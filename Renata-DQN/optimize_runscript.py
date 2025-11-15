######
# This script will serve as the optimization script for DQN hyperparameter tuning using Optuna
###### 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import os
import csv
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset
from itertools import combinations
from datetime import datetime
import optuna
import json

# Setup
project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The device being used is: {device}")

# Constants
NUCLEOTIDE_MAP = {"P": 0, "A": 1, "T": 2, "C": 3, "G": 4}
NUCLEOTIDES = {v: k for k, v in NUCLEOTIDE_MAP.items()}
GAP_CHAR = '-'  # for internal use during column construction

# Configuration
training_size = 1000
max_nt_num = 150
MAX_MSA_LEN = 50
MAX_N_SEQS = 3


def masked_epsilon_greedy(q_values: torch.Tensor, valid_mask: np.ndarray, epsilon: float, rng=None) -> int:
    if q_values.ndim > 1:
        q_values = q_values.reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    
    if rng is None:
        rng = np.random
    
    valid_idx = np.flatnonzero(valid_mask)
    
    if valid_idx.size == 0:
        return int(torch.argmax(q_values).item())
    
    if rng.random() < epsilon:
        return int(rng.choice(valid_idx))
    
    q = q_values.detach().cpu().numpy().copy()
    q[~valid_mask] = -np.inf
    return int(np.argmax(q))

def get_expected_alignment(sample):
    """
    Retrieve the expected alignment from the sample, if available.
    Checks for keys 'solution', 'aligned', and 'target' in that order.
    Returns a list of aligned sequences as strings, or None if not found.
    """
    for key in ("solution", "aligned", "target"):
        if key in sample and sample[key] is not None:
            result = sample[key]
            if isinstance(result, str):
                return [result]
            return [str(s) for s in result]
    return None

def get_expected_steps(sample, scale_factor=1.2, min_steps=10, max_steps=300):
    """
    Estimate expected number of alignment steps (columns) for the column-based environment.
    This approximates the final alignment length, without looking at the ground truth.
    """
    if 'start' in sample and isinstance(sample['start'], list):
        lengths = [len(seq) for seq in sample['start']]
        avg_len = np.mean(lengths)
        max_len = max(lengths)

        # Estimate final aligned length (a few gaps will expand it)
        est_len = int(scale_factor * max_len)
        return int(np.clip(est_len, min_steps, max_steps))

    # Fallback default because max length of sequences is 50 
    return 50

def get_expected_gaps(sample, max_factor=2.0, min_steps=1, max_steps_cap=1000):
    """
    Column-based budget: at most ~2 gaps per original column.
    No GT usage; only looks at raw starting sequences.
    """
    seqs = sample["start"]
    # print(f"withint get expected gaps, seqs: {seqs}")
    L = max(len(s) for s in seqs)
    steps = int(max_factor * L)          # e.g., 2 * longest raw length
    return max(min_steps, min(steps, max_steps_cap))

def _count_inserted_gaps_from_sequences(start, solution):
    dash_start = sum(str(s).count('-') for s in start)
    dash_solution = sum(str(s).count('-') for s in solution)
    return max(0, dash_solution - dash_start)

def convert_column_major_solution(msa_string, n_seq):
    """
    Converts a column-major MSA string (down columns first) into
    a row-major list of aligned sequences.
    
    Args:
        msa_string (str): e.g. "AAACC---CGGTTTT"
        n_seq (int): number of sequences (rows)
    
    Returns:
        list[str]: e.g. ["ACGT-", "A-GT-", "AC-T-"]
    """
    if not msa_string or n_seq <= 0:
        return []

    # Split into chunks of n_seq (each chunk = one column)
    columns = [msa_string[i:i+n_seq] for i in range(0, len(msa_string), n_seq)]

    # Transpose columns -> rows
    seqs = [''.join(col[i] for col in columns) for i in range(n_seq)]
    return seqs

def convert_huggingface_to_samples(dataset, max_samples=None):
    samples = []
    for i, ex in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        unaligned_seqs = ex.get('unaligned_seqs', {})
        MSA = ex.get('MSA', "")

        if not unaligned_seqs or not MSA:
            continue

        start = [unaligned_seqs[k] for k in sorted(unaligned_seqs.keys())]
        n_seq = len(start)
        solution = convert_column_major_solution(MSA, n_seq)

        accepted_pairs = [(str(a), str(b)) for a, b in combinations(range(len(start)), 2)]
        n_gaps = _count_inserted_gaps_from_sequences(start, solution)

        sample = {
            'start': start,
            'solution': solution,
            'n_gaps': n_gaps,
            'moves': [-1] * n_gaps,  # keep list length equal to n_gaps as this is never actually used in the DQN
            'n_sequences': len(start),
            'idx': i
        }
        samples.append(sample)
    return samples

def filter_by_seq_length(example, max_len=MAX_MSA_LEN):
    """Keep only samples where every unaligned sequence is <= max_len."""
    if "unaligned_seqs" not in example:
        return False
    seqs = example["unaligned_seqs"].values() if isinstance(example["unaligned_seqs"], dict) else example["unaligned_seqs"]
    return all(len(seq) <= max_len for seq in seqs)

def run_dqn_inference(agent, env, expected_gaps, max_steps=None):
    """
    Runs an inference episode using the trained DQN agent in the consensus-free environment.
    """
    if max_steps is None:
        max_steps = expected_gaps * 2

    state = env.reset()
    # env.randomize_sequence_order(apply=True)

    actions_taken = []
    valid_mask = env.get_valid_action_mask()

    for step in range(max_steps):
        if valid_mask.sum() == 0:
            break

        # Predict next action
        action = agent.predict(state, valid_action_mask=valid_mask)

        # Convert linear action index to (sequence, position)
        seq_idx = action // env.max_msa_len
        pos = action % env.max_msa_len
        actions_taken.append((seq_idx, pos))

        # Apply action in environment
        _, next_state, done = env.step(action)
        state = next_state
        valid_mask = env.get_valid_action_mask()

        if done == 1:
            break

    predicted = env.get_original_alignment()

    # Clean up padding ('P') — treat it as gap ('-') for clarity
    predicted = [seq.replace('P', '-') for seq in predicted[:env.original_n_seqs]]

    return predicted, actions_taken

class ReplayMemory:
    def __init__(self, memory_size=1000):
        self.storage = []
        self.memory_size = memory_size
        self.size = 0
    
    def store(self, data: tuple):
        if len(self.storage) == self.memory_size:
            self.storage.pop(0)
        self.storage.append(data)
        self.size = min(self.size + 1, self.memory_size)
    
    def sample(self, batch_size):
        samples = random.sample(self.storage, batch_size)
        state = [s for s, _, _, _, _, _ in samples]
        next_state = [ns for _, ns, _, _, _, _ in samples]
        action = [a for _, _, a, _, _, _ in samples]
        reward = [r for _, _, _, r, _, _ in samples]
        done = [d for _, _, _, _, d, _ in samples]
        next_mask = [m for _, _, _, _, _, m in samples]
        return state, next_state, action, reward, done, next_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.register_buffer("pos_table", self._get_sinusoid_table(n_position, d_hid))
    
    @staticmethod
    def _get_sinusoid_table(n_position, d_hid):
        positions = torch.arange(n_position).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-np.log(10000.0) / d_hid))
        sin = torch.sin(positions * div_term)
        cos = torch.cos(positions * div_term)
        return torch.cat([sin, cos], dim=-1).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1), :].clone().detach()

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, 1, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, 1, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, 1, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_model, n_position, d_k=164, d_v=164, pad_idx=0, dropout=0.1):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.self_attention = SelfAttention(d_model, d_k, d_v, dropout=dropout)
    
    def forward(self, src_seq, mask):
        enc_output = self.src_word_emb(src_seq)
        enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        enc_output, _ = self.self_attention(enc_output, enc_output, enc_output, mask=mask)
        return enc_output

class QNetwork(nn.Module):
    def __init__(self, num_sequences, max_sequence_length, num_actions, max_action_value, d_model=64):
        super().__init__()
        self.num_sequences = num_sequences
        self.num_rows = num_sequences  # <-- no +2

        dim = self.num_rows * max_sequence_length
        # Encoder: your existing module that maps (B, dim) with an attention/MLP stack
        self.encoder = Encoder(5, d_model, dim)                                                             ####### Changed from 6 to 5 to get rid of gap since column based will never encounter gaps in the embedding stage

        # Learnable row embeddings (still helpful)
        self.seq_embedding = nn.Embedding(self.num_rows, d_model)
        nn.init.normal_(self.seq_embedding.weight, 0.0, 0.1)

        self.fc1 = nn.Linear(dim * d_model, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, R, C = x.shape          # R == num_sequences
        x_flat = x.view(B, R * C)
        mask_flat = (x_flat != 0).unsqueeze(1)

        h = self.encoder(x_flat, mask_flat)   # shape can be (B, R*C, d_model) or (B, d_model) per your Encoder
        if h.dim() == 3:
            h = h.view(B, R, C, -1)

        # add per-row embedding (broadcast)
        row_ids = torch.arange(R, device=h.device)
        row_emb = self.seq_embedding(row_ids)[None, :, None, :]  # (1,R,1,d_model)
        h = h + row_emb

        h = h.reshape(B, -1)
        h = F.leaky_relu(self.fc1(h)); h = self.dropout(h)
        h = F.leaky_relu(self.fc2(h)); h = self.dropout(h)
        q = self.fc3(h)
        return q

class DQNAgent:    
    def __init__(self, action_number, num_seqs, max_grid, max_value,
                 epsilon=0.8, delta=0.05, decrement_iteration=5,
                 update_iteration=128, batch_size=128, gamma=1.0,
                 learning_rate=0.001, memory_size=1000):
        self.seq_num = num_seqs
        self.max_seq_len = max_grid  # <-- not +1
        self.action_number = action_number

        self.eval_net = QNetwork(num_seqs, self.max_seq_len, action_number, max_value).to(device)
        self.target_net = QNetwork(num_seqs, self.max_seq_len, action_number, max_value).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.replay_memory = ReplayMemory(memory_size=memory_size)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.SmoothL1Loss()
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_iteration = update_iteration
        self.update_step_counter = 0
        self.tau = 0.005
        self.use_double_dqn = True
        
        self.initial_epsilon = epsilon
        self.current_epsilon = epsilon
        self.epsilon_end = delta
        self.epsilon_decay = 0.999
        
        self.losses = []
        self.epsilons = []

    def update_epsilon(self):
        if self.update_step_counter < 5000:
            decay_rate = 0.9999
        elif self.update_step_counter < 10000:
            decay_rate = 0.999
        else:
            decay_rate = 0.995
        
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon * decay_rate)
        self.epsilons.append(self.current_epsilon)

    def select_action(self, state, valid_action_mask=None):
        is_random = (random.random() <= self.current_epsilon)
        
        if is_random:
            if valid_action_mask is not None:
                valid_idx = np.flatnonzero(valid_action_mask)
                action = int(np.random.choice(valid_idx)) if len(valid_idx) else random.randrange(self.action_number)
            else:
                action = random.randrange(self.action_number)
        else:
            self.eval_net.eval()
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.long, device=device).view(1, self.seq_num, self.max_seq_len)
                q = self.eval_net(s).squeeze(0).detach().cpu().numpy()
            self.eval_net.train()
            
            if valid_action_mask is not None:
                q[~valid_action_mask] = -np.inf
            
            action = int(np.argmax(q))
        
        return action

    def predict(self, state, valid_action_mask=None):
        self.eval_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.long, device=device).view(1, self.seq_num, self.max_seq_len)
            q = self.eval_net(s).squeeze(0).detach().cpu().numpy()
        self.eval_net.train()
        
        if valid_action_mask is not None:
            q[~valid_action_mask] = -np.inf
        
        return int(np.nanargmax(q))

    def forward(self, state):
        self.eval_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.long, device=device).view(
                1, self.seq_num, self.max_seq_len
            )
            q = self.eval_net(s).squeeze(0)
        self.eval_net.train()
        return q
    
    @property
    def epsilon(self):
        return self.current_epsilon

    def update(self):
        self.update_step_counter += 1
        
        if self.replay_memory.size < self.batch_size:
            return None
        
        state, next_state, action, reward, done, next_mask = self.replay_memory.sample(self.batch_size)

        batch_state = torch.LongTensor(state).to(device).view(-1, self.seq_num, self.max_seq_len)
        batch_next_state = torch.LongTensor(next_state).to(device).view(-1, self.seq_num, self.max_seq_len)
        batch_action = torch.LongTensor(action).unsqueeze(-1).to(device)
        batch_reward = torch.FloatTensor(reward).unsqueeze(-1).to(device)
        batch_done = torch.FloatTensor(done).unsqueeze(-1).to(device)
        batch_next_mask = torch.BoolTensor(next_mask).to(device)
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        
        with torch.no_grad():
            if self.use_double_dqn:
                q_next_online = self.eval_net(batch_next_state)
                q_next_online_masked = q_next_online.clone()
                q_next_online_masked[~batch_next_mask] = float('-inf')
                best_next_actions = q_next_online_masked.max(1)[1]
                
                q_next_target = self.target_net(batch_next_state)
                q_next = q_next_target.gather(1, best_next_actions.unsqueeze(1))
            else:
                q_next_target = self.target_net(batch_next_state)
                q_next_masked = q_next_target.clone()
                q_next_masked[~batch_next_mask] = float('-inf')
                q_next = q_next_masked.max(1)[0].unsqueeze(-1)
            
            q_next = torch.where(torch.isinf(q_next), torch.zeros_like(q_next), q_next)
            q_target = batch_reward + (1.0 - batch_done) * self.gamma * q_next
            q_target = torch.clamp(q_target, -10.0, 10.0)
        
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * online_param.data)
        
        self.losses.append(loss.item())
        return loss.item()
    
    def save_model(self, path):
        """Save the current model (eval_net) and target_net weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.current_epsilon,
            'update_step_counter': self.update_step_counter
        }, path)

    def load_model(self, path, map_location=None):
        """Load model weights from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=map_location or device)
        self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epsilon = checkpoint.get('epsilon', self.current_epsilon)
        self.update_step_counter = checkpoint.get('update_step_counter', 0)

class AlignmentEnvironment:
    def __init__(self, sequences, total_gap, max_n_seqs=None, max_msa_len=None):
        """
        Column-based alignment environment.
        At each step, the agent decides which sequences get a gap in the next aligned column.
        """
        self.original_n_seqs = len(sequences)
        self.max_n_seqs = max_n_seqs or len(sequences)
        self.max_msa_len = max_msa_len or max(len(s) for s in sequences)

        # Pad to equal length
        sequences = self._pad_sequences(sequences, self.max_msa_len)
        while len(sequences) < self.max_n_seqs:
            sequences.append("A" * self.max_msa_len)  # dummy padding row, won't affect alignment

        self.initial_sequences = [list(s) for s in sequences]
        self.total_gap = total_gap
        self.initial_gap = total_gap
        self.reset()

    @staticmethod
    def _pad_sequences(sequences, target_len):
        return [s.ljust(target_len, "P") for s in sequences]       # we can do another nucleotide for padding but not sure if we would want that for the embedding ???

    # -------------------- ACTION SPACE --------------------
    def get_valid_action_mask(self):
        n = self.original_n_seqs
        n_actions = 2 ** n
        mask = np.ones(n_actions, dtype=bool)

        # Ban the all-gap action (every bit = 1)
        mask[n_actions - 1] = False
        return mask


    # ---------- CONSENSUS-FREE REWARD ----------
    @staticmethod
    def _pairwise_column_score(rows, match_reward=2.0, mismatch_penalty=-2.0, gap_penalty=-1.0):
        """
        Computes the pairwise sum of scores for the given columns.
        Each column is compared across all pairs of sequences.
        
        - Matches: +match_reward
        - Mismatches: +mismatch_penalty
        - Gaps ('-' or 'P'): +gap_penalty
        
        Args:
            rows: List[List[str]] where rows[r][c] = nucleotide or gap symbol.
        """
        if not rows:
            return 0.0
        
        n = len(rows)
        L = len(rows[0])
        total_score = 0.0

        for c in range(L):
            # Take column across all sequences
            col = [rows[r][c] for r in range(n)]
            # Treat 'P' as '-'
            col = ['-' if x == 'P' else x for x in col]

            # Compare all sequence pairs in this column
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = col[i], col[j]
                    if a == '-' or b == '-':
                        total_score += gap_penalty
                    elif a == b:
                        total_score += match_reward
                    else:
                        total_score += mismatch_penalty

        return float(total_score)


    def get_current_state(self):
        # pad each unaligned row to max_msa_len with 'P'
        rows = []
        for i in range(self.max_n_seqs):
            rem = self.unaligned[i]
            if len(rem) >= self.max_msa_len:
                row = rem[:self.max_msa_len]
            else:
                row = rem + ['P'] * (self.max_msa_len - len(rem))
            rows.append([NUCLEOTIDE_MAP.get(x, 0) for x in row])
        # flatten R x C
        flat = []
        for r in rows:
            flat.extend(r)
        return flat

    # Predicted-only metrics (SP/CS) for validation & optional reward shaping
    def _predicted_sp_score(self, msa):
        # +2 match (non-gap), -1 if any gap, -2 mismatch (non-gap)
        n = len(msa)
        if n == 0: return 0.0
        L = len(msa[0])
        assert all(len(s) == L for s in msa)
        score = 0
        from itertools import combinations
        for i, j in combinations(range(n), 2):
            si, sj = msa[i], msa[j]
            for c in range(L):
                a, b = si[c], sj[c]
                if a == '-' or b == '-':
                    score -= 1
                elif a == b:
                    score += 2
                else:
                    score -= 2
        return float(score)

    def _predicted_cs_fraction(self, msa):
        n = len(msa)
        if n == 0: return 0.0
        L = len(msa[0])
        assert all(len(s) == L for s in msa)
        good = 0
        for c in range(L):
            col = [msa[r][c] for r in range(n)]
            nz = [x for x in col if x != '-']
            if nz and len(set(nz)) == 1:
                good += 1
        return good / L if L > 0 else 0.0

    # Public calc for validation “reward-like” score
    def calc_reward_from_alignment(self, aligned_list_of_str):
        rows = [[ch for ch in row] for row in aligned_list_of_str[:self.original_n_seqs]]
        return self._pairwise_column_score(rows)

    def calc_reward(self):
        """Full alignment score of current state."""
        rows = ["".join(r) for r in self.aligned[:self.original_n_seqs]]
        return self._pairwise_column_score([[ch for ch in row] for row in rows])

    # -------------------- STEP --------------------
    def step(self, action):
        """
        Execute one column alignment step.
        Action encodes which sequences receive a gap this round.
        """
        if self.done_flag:
            return 0.0, self.get_current_state(), 1

        n = self.original_n_seqs
        bin_mask = [int(x) for x in bin(action)[2:].zfill(n)]  # e.g., 0b101 -> [1,0,1]

        for i in range(n):
            if bin_mask[i] == 1:
                # insert gap
                self.aligned[i].append('-')
                self.total_gap -= 1
            else:
                # consume next nucleotide if available
                if self.unaligned[i]:
                    nt = self.unaligned[i].pop(0)
                    self.aligned[i].append(nt)
                else:
                    # no bases left, add gap
                    self.aligned[i].append('-')
                    self.total_gap -= 1

        # compute dense step reward
        last_col_rows = [[self.aligned[i][-1]] for i in range(n)]  # n rows, length 1
        reward = self._pairwise_column_score(last_col_rows)
        reward = float(np.clip(reward, -5.0, 5.0))

        # termination condition: all unaligned parts are empty or contain only 'P'
        def _is_exhausted(seq):
            return all(ch == 'P' for ch in seq) or len(seq) == 0
        self.done_flag = all(_is_exhausted(u) for u in self.unaligned[:n])
        done = int(self.done_flag)

        return reward, self.get_current_state(), done

    # -------- Reset / Permutation (unchanged) --------
    def get_alignment(self):
        alignment = []
        for i in range(len(self.aligned)):
            alignment.append(''.join([NUCLEOTIDES[self.aligned[i][j]] for j in range(len(self.aligned[i]))]))
        return alignment

    def get_original_alignment(self):
        max_len = max(len(r) for r in self.aligned)
        out = []
        for i in range(self.original_n_seqs):
            row = self.aligned[i] + ['-'] * (max_len - len(self.aligned[i]))
            out.append(''.join(row))
        return out


    # -------------------- STATE / RESET --------------------
    def reset(self):
        self.unaligned = [list(seq) for seq in self.initial_sequences]
        self.aligned = [[] for _ in range(self.max_n_seqs)]
        self.total_gap = self.initial_gap
        self.done_flag = False
        return self.get_current_state()

    def randomize_sequence_order(self, apply=True):
        if not apply or self.original_n_seqs <= 1:
            self.seq_permutation = None
            self.seq_permutation_inv = None
            return
        self.seq_permutation = np.random.permutation(self.original_n_seqs)
        self.seq_permutation_inv = np.argsort(self.seq_permutation)
        self.original[:self.original_n_seqs]        = [self.original[i]        for i in self.seq_permutation]
        self.sequences[:self.original_n_seqs]       = [self.sequences[i]       for i in self.seq_permutation]
        self.sep_nuc_in_seq[:self.original_n_seqs]  = [self.sep_nuc_in_seq[i]  for i in self.seq_permutation]
        self.label_encoded_seqs[:self.original_n_seqs] = [self.label_encoded_seqs[i] for i in self.seq_permutation]
        gaps_copy = self.gaps_per_sequence[:self.original_n_seqs].copy()
        for new_idx, old_idx in enumerate(self.seq_permutation):
            self.gaps_per_sequence[new_idx] = gaps_copy[old_idx]

    
    def get_unpermuted_gaps(self):
        if self.seq_permutation_inv is None:
            return self.gaps_per_sequence[:self.original_n_seqs]
        
        original_order = [0] * self.original_n_seqs
        for new_idx, old_idx in enumerate(self.seq_permutation):
            original_order[old_idx] = self.gaps_per_sequence[new_idx]
        return original_order
    
    @staticmethod
    def get_sp_score(pred, match=2, mismatch=-2, gap = -1):
        n = len(pred)
        L = len(pred[0])

        score = 0
        for i, j in combinations(range(n), 2):
            si, sj = pred[i], pred[j]
            for c in range(L):
                a, b = si[c], sj[c]
                if a == '-' and b == '-':
                    score += 0 
                elif a == '-' or b == '-':
                    score += gap
                elif a == b:
                    score += match
                else:
                    score += mismatch
        return float(score)
    
    @staticmethod
    def get_cs_score(pred):
        n = len(pred)
        L = len(pred[0])

        good = 0
        for c in range(L):
            col = [pred[r][c] for r in range(n)]
            nz = [x for x in col if x != '-']      # non-gap residues
            if len(nz) == 0:
                # no residues in this column → by definition not a “matching residues” column
                continue
            if len(set(nz)) == 1:
                good += 1
        return good / L if L > 0 else 0.0
    
    @staticmethod
    def compute_alignment_metrics(pred, ref):
        """
        Computes predicted-only scores (SP, CS) and reference-based metrics (Q*, TC*).

        Q(A,R): pair-based accuracy (sum-of-pairs)
        TC(A,R): column-based accuracy (total-column match)

        Columns are compared by tuples (bases_across_sequences, column_index)
        so the logic is parallel to the pair-based comparison.
        """
        assert len(pred) == len(ref), "Pred/ref must have same number of sequences."
        n = len(ref)
        if n == 0:
            return {k: 0.0 for k in [
                'pred_sp','pred_cs','Q_acc','Q_prec','Q_rec','Q_f1',
                'TC_acc','TC_prec','TC_rec','TC_f1'
            ]}
        Lp = len(pred[0])
        Lr = len(ref[0])
        assert all(len(s) == Lp for s in pred)
        assert all(len(s) == Lr for s in ref)
        L = min(Lp, Lr)

        # --- predicted-only metrics
        pred_sp = AlignmentEnvironment.get_sp_score(pred)
        pred_cs = AlignmentEnvironment.get_cs_score(pred)

        # ---------- Q (pair) metrics ----------
        def get_pairs(msa):
            pairs = set()
            for i, j in combinations(range(n), 2):
                seq_i, seq_j = msa[i], msa[j]
                for c in range(L):
                    a, b = seq_i[c], seq_j[c]
                    if a != '-' and b != '-':
                        pairs.add((i, j, c))
            return pairs

        pred_pairs = get_pairs(pred)
        ref_pairs = get_pairs(ref)

        TPp = len(pred_pairs & ref_pairs)
        FPp = len(pred_pairs - ref_pairs)
        FNp = len(ref_pairs - pred_pairs)

        Q_acc  = TPp / len(ref_pairs) if len(ref_pairs) > 0 else 0.0
        Q_prec = TPp / (TPp + FPp) if (TPp + FPp) > 0 else 0.0
        Q_rec  = TPp / (TPp + FNp) if (TPp + FNp) > 0 else 0.0
        Q_f1   = (2 * Q_prec * Q_rec / (Q_prec + Q_rec)) if (Q_prec + Q_rec) > 0 else 0.0

        # ---------- TC (column) metrics ----------
        def get_columns(msa):
            cols = set()
            for c in range(L):
                col = tuple(msa[r][c] for r in range(n))
                cols.add((c, col))  # include index for uniqueness
            return cols

        pred_cols = get_columns(pred)
        ref_cols  = get_columns(ref)

        TPc = len(pred_cols & ref_cols)
        FPc = len(pred_cols - ref_cols)
        FNc = len(ref_cols - pred_cols)

        TC_acc  = TPc / len(ref_cols) if len(ref_cols) > 0 else 0.0
        TC_prec = TPc / (TPc + FPc) if (TPc + FPc) > 0 else 0.0
        TC_rec  = TPc / (TPc + FNc) if (TPc + FNc) > 0 else 0.0
        TC_f1   = (2 * TC_prec * TC_rec / (TC_prec + TC_rec)) if (TC_prec + TC_rec) > 0 else 0.0

        return {
            "pred_sp": float(pred_sp),
            "pred_cs": float(pred_cs),
            "Q_acc": float(Q_acc),
            "Q_prec": float(Q_prec),
            "Q_rec": float(Q_rec),
            "Q_f1": float(Q_f1),
            "TC_acc": float(TC_acc),
            "TC_prec": float(TC_prec),
            "TC_rec": float(TC_rec),
            "TC_f1": float(TC_f1)
        }

####################################################################################################
#### Optuna Optimization ####
####################################################################################################
def suggest_hyperparams(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "gamma": trial.suggest_uniform("gamma", 0.90, 0.999),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "epsilon_start": trial.suggest_uniform("epsilon_start", 0.8, 1.0),
        "epsilon_end": trial.suggest_uniform("epsilon_end", 0.01, 0.2),
        "epsilon_decay": trial.suggest_uniform("epsilon_decay", 0.995, 0.99999),
        # "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        # "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        # "target_update_freq": trial.suggest_categorical("target_update_freq", [100, 250, 500]),
    }

# --- Load and filter datasets ---
ds = load_dataset("dotan1111/MSA-nuc-3-seq", split="train")
ds = ds.filter(filter_by_seq_length)
train_samples = convert_huggingface_to_samples(ds, max_samples=training_size)

ds = load_dataset("dotan1111/MSA-nuc-3-seq", split="validation")
ds = ds.filter(filter_by_seq_length)
val_samples = convert_huggingface_to_samples(ds)

ds = load_dataset("dotan1111/MSA-nuc-3-seq", split="test")
ds = ds.filter(filter_by_seq_length)
test_samples = convert_huggingface_to_samples(ds) 

# --- Define Optuna Objective and get the top 5 trials ---
def objective(trial):

    # ----------- HYPERPARAMETERS -----------
    hp = suggest_hyperparams(trial)

    # ---------- BUILD AGENT ----------
    agent = DQNAgent(
        action_number = 2 ** MAX_N_SEQS,
        num_seqs = MAX_N_SEQS,
        max_grid = MAX_MSA_LEN,
        max_value = MAX_MSA_LEN * 100,
        epsilon = hp["epsilon_start"],
        delta = hp["epsilon_end"],
        batch_size = hp["batch_size"],
        gamma = hp["gamma"],
        learning_rate = hp["learning_rate"],
        memory_size = 5000,
        # hidden_size = hp["hidden_size"],
        # n_heads = hp["n_heads"],
        # target_update_freq = hp["target_update_freq"]
    )

    # ----------- TRAINING SETTINGS ----------
    episodes_per_trial = 200  # ~10% of full training
    samples = random.sample(train_samples, episodes_per_trial)

    reward_window = []   # for Optuna pruning :(

    # --------------- MINI TRAINING LOOP -----------------
    for ep, sample in enumerate(samples):

        env = AlignmentEnvironment(
            sequences = sample["start"],
            total_gap = get_expected_gaps(sample),
            max_n_seqs = MAX_N_SEQS,
            max_msa_len = MAX_MSA_LEN
        )

        state = env.reset()
        valid_mask = env.get_valid_action_mask()
        episode_reward = 0

        max_steps = get_expected_gaps(sample)

        for t in range(max_steps):
            action = agent.select_action(state, valid_action_mask=valid_mask)
            reward, next_state, done = env.step(action)
            next_mask = env.get_valid_action_mask()

            agent.replay_memory.store((state, next_state, action, reward, done, next_mask))
            agent.update()

            state = next_state
            valid_mask = next_mask
            episode_reward += reward

            if done:
                break

        agent.update_epsilon()
        reward_window.append(episode_reward)

        # ------- Optuna pruning: stop bad trials early -------
        if ep % 20 == 0:
            avg_recent = np.mean(reward_window[-20:])
            trial.report(avg_recent, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # ---------------- VALIDATION ----------------
    # Evaluate on 50 validation MSAs
    val_scores = []
    val_cases = random.sample(val_samples, 50)

    for val_sample in val_cases:
        env = AlignmentEnvironment(
            sequences = val_sample["start"],
            total_gap = get_expected_gaps(val_sample),
            max_n_seqs = MAX_N_SEQS,
            max_msa_len = MAX_MSA_LEN
        )
        predicted, _ = run_dqn_inference(agent, env, expected_gaps=get_expected_gaps(val_sample))
        metrics = env.compute_alignment_metrics(predicted, val_sample["solution"])

        val_scores.append(metrics["pred_sp"])   # I think this is a good starting point 

    return float(np.mean(val_scores))

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
)

study.optimize(objective, n_trials=60)

print("Best Trial:", study.best_trial.params)

# lets have a look at the top 5 trials and retrain them fully
best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
run_dir = "../result/full_runs/"
os.makedirs(run_dir, exist_ok=True)

# Convert trials into a clean serializable list
top5_data = []
for t in best_trials:
    top5_data.append({
        "trial_id": t.number,
        "value": t.value,
        "params": t.params,
        "state": str(t.state),
    })
# Save to file
top5_path = os.path.join(run_dir, "top5_hyperparams.json")
with open(top5_path, "w") as f:
    json.dump(top5_data, f, indent=4)


# --- Full retraining function ---
def train_full_model(index, hparams, 
                     train_samples=train_samples, 
                     val_samples=val_samples, 
                     max_epochs=20,
                     episodes_per_epoch=200,
                     val_per_epoch=50):

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"../result/full_runs/"
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, f"{index}_train_log.csv")
    seq_path = os.path.join(run_dir, f"{index}_sequence_log.csv")
    model_path = os.path.join(run_dir, f"{index}_model.pt")

    # ---------------------- CSV HEADERS ----------------------
    metric_fields = [
        "epoch",
        "pred_sp", "pred_cs",
        "Q_acc", "Q_prec", "Q_rec", "Q_f1",
        "TC_acc", "TC_prec", "TC_rec", "TC_f1"
    ]

    seq_fields = ["epoch", "sample_index", "unaligned_seqs", "predicted_alignment"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=metric_fields).writeheader()

    with open(seq_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=seq_fields).writeheader()

    # ---------------------- BUILD AGENT ----------------------
    agent = DQNAgent(
        action_number = 2 ** MAX_N_SEQS,
        num_seqs = MAX_N_SEQS,
        max_grid = MAX_MSA_LEN,
        max_value = MAX_MSA_LEN * 100,
        epsilon = hparams["epsilon_start"],
        delta = hparams["epsilon_end"],
        batch_size = hparams["batch_size"],
        gamma = hparams["gamma"],
        learning_rate = hparams["learning_rate"],
        memory_size = 5000
    )

    # ---------------------- TRAINING LOOP ----------------------
    for epoch in range(1, max_epochs + 1):

        episode_samples = random.sample(train_samples, episodes_per_epoch)
        for sample in episode_samples:

            env = AlignmentEnvironment(
                sequences=sample["start"],
                total_gap=get_expected_gaps(sample),
                max_n_seqs=MAX_N_SEQS,
                max_msa_len=MAX_MSA_LEN
            )
            state = env.reset()
            valid_mask = env.get_valid_action_mask()
            max_steps = get_expected_gaps(sample)

            for t in range(max_steps):
                action = agent.select_action(state, valid_action_mask=valid_mask)
                reward, next_state, done = env.step(action)
                next_mask = env.get_valid_action_mask()

                agent.replay_memory.store((state, next_state, action, reward, done, next_mask))
                agent.update()

                state = next_state
                valid_mask = next_mask
                if done:
                    break

            agent.update_epsilon()

        # ---------------------- VALIDATION ----------------------
        val_cases = random.sample(val_samples, val_per_epoch)
        aggregated = {k: [] for k in metric_fields if k != "epoch"}

        for idx, val_sample in enumerate(val_cases):

            env = AlignmentEnvironment(
                sequences=val_sample["start"],
                total_gap=get_expected_gaps(val_sample),
                max_n_seqs=MAX_N_SEQS,
                max_msa_len=MAX_MSA_LEN
            )

            predicted, _ = run_dqn_inference(agent, env, get_expected_gaps(val_sample))
            metrics = env.compute_alignment_metrics(predicted, val_sample["solution"])

            for k in aggregated:
                aggregated[k].append(metrics[k])

            # ------------------ OUTPUT ALIGNMENT LOG ------------------
            with open(seq_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=seq_fields)
                writer.writerow({
                    "epoch": epoch,
                    "sample_index": idx,
                    "unaligned_seqs": " || ".join(val_sample["start"]),
                    "predicted_alignment": " || ".join(predicted)
                })

        # ---------------------- WRITE METRICS ----------------------
        avg_metrics = {k: float(np.mean(v)) for k, v in aggregated.items()}

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writerow({
                "epoch": epoch,
                **avg_metrics
            })

        print(f"Epoch {epoch} validation SP={avg_metrics['pred_sp']:.2f}, Q_acc={avg_metrics['Q_acc']:.3f}")

    # ---------------------- SAVE MODEL ----------------------
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    print(f"Metrics log saved to {log_path}")
    print(f"Sequence log saved to {seq_path}")

    return avg_metrics["pred_sp"]

for i, trial in enumerate(best_trials):
    print(f"Retraining best trial {i} with params: {trial.params}")
    final_sp = train_full_model(i, trial.params)
    print(f"Trial {i} final validation SP: {final_sp:.2f}")


