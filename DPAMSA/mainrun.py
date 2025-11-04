import sys, os, random, math, json
import numpy as np
import torch
from tqdm import tqdm
from env import Environment
from dqn import DQN
import config
from MSADataset import MSADataset


# ---------- helper func ----------
def set_seeds(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_agent_from_example(seqs):
    # print(f"The length of the sequences are {[len(s) for s in seqs]}")
    env = Environment(seqs)
    #env.padding()
    agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    return agent

def reshape_ref_alignment(ref_aln: str, n_seq: int):
    """Convert flattened column-major reference alignment to list of per-sequence strings."""
    assert len(ref_aln) % n_seq == 0, (
        f"Reference alignment length {len(ref_aln)} must be divisible by number of sequences {n_seq}"
    )
    L_ref = len(ref_aln) // n_seq
    # For each sequence, collect positions r, r+n_seq, r+2*n_seq, ...
    ref_lines = ["".join(ref_aln[r + c * n_seq] for c in range(L_ref)) for r in range(n_seq)]
    return ref_lines

def calc_ref_PS(ref_aln: str, n_seq: int):
    """
    Compute the Sum-of-Pairs (SP) score for the reference alignment.
    Uses the same scoring scheme as environment.calc_score().
    """
    seqs = reshape_ref_alignment(ref_aln, n_seq)
    n_cols = len(seqs[0])
    score = 0

    for i in range(n_cols):
        for j in range(n_seq):
            for k in range(j + 1, n_seq):
                a, b = seqs[j][i], seqs[k][i]
                if a == '-' and b == '-':
                    # No score for gap-gap
                    score += 0
                    #print(f"[DEBUG] No score for gap-gap in column {i}, seqs {j} and {k}: {a}, {b}")
                elif a == '-' or b == '-':
                    score += config.GAP_PENALTY
                    #print(f"[DEBUG] Gap penalty applied for column {i}, seqs {j} and {k}: {a}, {b}")
                elif a == b:
                    score += config.MATCH_REWARD
                    #print(f"[DEBUG] Match reward applied for column {i}, seqs {j} and {k}: {a}, {b}")
                else:
                    score += config.MISMATCH_PENALTY
                    #print(f"[DEBUG] Mismatch penalty applied for column {i}, seqs {j} and {k}: {a}, {b}")

    return score

def calc_ref_CS(ref_aln: str, n_seq: int):
    """
    Compute the Column Score (CS) for the reference alignment.
    Returns the fraction of columns where all residues match (excluding gaps).
    """
    seqs = reshape_ref_alignment(ref_aln, n_seq)
    n_cols = len(seqs[0])
    exact_match_count = 0

    for i in range(n_cols):
        col = [seqs[r][i] for r in range(n_seq)]
        if all(c == col[0] for c in col):  # all same base (including gaps)
            exact_match_count += 1

    cs = exact_match_count / n_cols if n_cols > 0 else 0
    return cs

def colwise_msa_metrics(pred_aln: str, ref_aln: str):
    """
    Evaluate multiple sequence alignment quality using column-wise precision/recall/F1.

    pred_aln: multiline string where each line is a predicted aligned sequence.
              Example:
                  A-C
                  AG-
                  ACC
    ref_aln:  flattened column-major reference string (e.g. 'AAA--CCGC')
              which encodes the same number of sequences (n_seq).
    """
    # --- Parse predicted alignment ---
    pred_lines = [ln.strip() for ln in pred_aln.strip().split("\n") if ln.strip()]
    n_seq = len(pred_lines)
    assert n_seq > 1, "Need at least 2 sequences"

    L_pred = len(pred_lines[0])
    assert all(len(ln) == L_pred for ln in pred_lines), "Predicted sequences must be same length"

    # Convert predicted lines into column tuples [(A,A,A), (-,-,G), (C,C,C), ...]
    pred_cols = [tuple(pred_lines[row][col] for row in range(n_seq)) for col in range(L_pred)]

    # --- Parse reference alignment (already column-major) ---
    assert len(ref_aln) % n_seq == 0, (
        f"Reference alignment length ({len(ref_aln)}) must be divisible by number of sequences ({n_seq})"
    )
    L_ref = len(ref_aln) // n_seq
    ref_cols = [tuple(ref_aln[c * n_seq + r] for r in range(n_seq)) for c in range(L_ref)]

    #print(f"[DEBUG] pred lines is: {pred_lines}")
    #print(f"[DEBUG] pred cols is: {pred_cols}")
    #print(f"[DEBUG] ref cols is: {ref_cols}")

    # --- Compute column-based matching ---
    min_len = min(len(pred_cols), len(ref_cols))
    TP = sum(pred_cols[i] == ref_cols[i] for i in range(min_len))
    FP = len(pred_cols) - TP
    FN = len(ref_cols) - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = TP / (TP + FP + FN + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "n_cols_pred": L_pred,
        "n_cols_ref": L_ref,
        "msa_PS": calc_ref_PS(ref_aln, n_seq),
        "msa_CS": calc_ref_CS(ref_aln, n_seq)
    }

# ---------- Evaluation helpers ----------
def eval_one(agent, seqs, ref_msa=None, use_greedy=True):
    """Evaluate one alignment episode and compute SP, CS, and optional F1."""
    env = Environment(seqs)
    state = env.reset()
    step_fn = agent.predict if use_greedy else agent.select

    while True:
        action = step_fn(state)
        _, next_state, done = env.step(action)
        state = next_state
        if done == 0:
            break

    env.padding()
    sp = env.calc_score()
    em = env.calc_exact_matched()
    cs = em / max(1, len(env.aligned[0]))
    pred_aln = env.get_alignment()

    metrics = {"SP": sp, "CS": cs}
    if ref_msa is not None:
        metrics.update(colwise_msa_metrics(pred_aln, ref_msa))
    return metrics


def eval_split(agent, dataset, n=200):
    """Run evaluation on n random samples from dataset."""
    results = []
    for _ in range(n):
        seqs, ref_msa = dataset.sample_row()
        m = eval_one(agent, seqs, ref_msa)
        results.append(m)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return avg

# ---------- Main training ----------
def train_multitask(train_ds, val_ds, ckpt_dir=config.weight_path, log_dir=config.score_path):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_seeds(42)

    # bootstrap agent from one sample row (to derive dimensions)
    seqs, msa_ref = train_ds.sample_row()
    agent = make_agent_from_example(seqs)
    best_val_cs = -1.0
    best_val_f1 = -1.0

    # Prepare training log
    log_path = os.path.join(log_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("episode,epsilon,SP,CS,accuracy,precision,recall,F1\n")

    pbar = tqdm(range(1, config.max_episode + 1), desc="train")
    for ep in pbar:
        print(f"\n=== Episode {ep} ===")
        # ----- sample one MSA (one row) and run an episode -----
        seqs, msa_ref = train_ds.sample_row()
        print(f"The length of the sequences within the pbar are {[len(s) for s in seqs]}")
        env = Environment(seqs)
        state = env.reset()
        ep_reward = 0

        while True:
            action = agent.select(state)  # epsilon-greedy
            reward, next_state, done = env.step(action)
            agent.replay_memory.push((state, next_state, action, reward, done))
            agent.update()
            ep_reward += reward
            if done == 0:
                break
            state = next_state

        agent.update_epsilon()

        # ----- periodic evaluation -----
        if ep % config.eval_every == 0 or ep == 1:
            val_metrics = eval_split(agent, val_ds, n=config.val_samples)
            msg = (f"[VAL] ep={ep}  SP={val_metrics['SP']:.2f}  "
                   f"CS={val_metrics['CS']:.3f}  F1={val_metrics['f1']:.3f}  "
                   f"eps={agent.current_epsilon:.3f}")
            print("\n" + msg)

            # Save metrics to log
            with open(log_path, "a") as f:
                f.write(f"{ep},{agent.current_epsilon:.4f},{val_metrics['SP']:.3f},{val_metrics['CS']:.3f},"
                        f"{val_metrics['accuracy']:.3f},{val_metrics['precision']:.3f},"
                        f"{val_metrics['recall']:.3f},{val_metrics['f1']:.3f},{val_metrics['msa_PS']:.3f},{val_metrics['msa_CS']:.3f}\n")

            # Save best models
            if val_metrics["CS"] > best_val_cs:
                best_val_cs = val_metrics["CS"]
                agent.save(f"best_cs_ep{ep}", ckpt_dir)
                with open(os.path.join(ckpt_dir, "best_cs_meta.json"), "w") as f:
                    json.dump({"episode": ep, "val_metrics": val_metrics}, f)

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                agent.save(f"best_f1_ep{ep}", ckpt_dir)
                with open(os.path.join(ckpt_dir, "best_f1_meta.json"), "w") as f:
                    json.dump({"episode": ep, "val_metrics": val_metrics}, f)

        # ----- periodic checkpoint -----
        if ep % config.save_every == 0:
            agent.save(f"shared_dqn_ep{ep}", ckpt_dir)

        # Update tqdm
        if ep % 200 == 0:
            pbar.set_postfix(eps=f"{agent.current_epsilon:.3f}",
                             lastR=f"{ep_reward:.1f}",
                             mem=agent.replay_memory.size)

    agent.save("shared_dqn_final", ckpt_dir)
    print(f"Training complete. Logs saved to {log_path}")
    return agent


# ---------- Testing ----------
def test_model(agent, test_ds, log_dir=config.score_path):
    """Evaluate a trained agent on the test dataset and save results."""
    os.makedirs(log_dir, exist_ok=True)
    test_log = os.path.join(log_dir, "testing_log.csv")

    results = []
    pbar = tqdm(range(config.test_samples), desc="testing")
    for _ in pbar:
        seqs, ref_msa = test_ds.sample_row()
        m = eval_one(agent, seqs, ref_msa)
        results.append(m)
        pbar.set_postfix(SP=f"{m['SP']:.1f}", CS=f"{m['CS']:.3f}", F1=f"{m['f1']:.3f}")

    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print(f"\n[TEST]  SP={avg['SP']:.2f}  CS={avg['CS']:.3f}  "
          f"Acc={avg['accuracy']:.3f}  F1={avg['f1']:.3f}")

    # Write CSV
    with open(test_log, "w") as f:
        f.write("index,SP,CS,accuracy,precision,recall,F1\n")
        for i, m in enumerate(results):
            f.write(f"{i},{m['SP']:.3f},{m['CS']:.3f},{m['accuracy']:.3f},"
                    f"{m['precision']:.3f},{m['recall']:.3f},{m['f1']:.3f}\n")
        f.write(f"\n# AVG,{avg['SP']:.3f},{avg['CS']:.3f},{avg['accuracy']:.3f},"
                f"{avg['precision']:.3f},{avg['recall']:.3f},{avg['f1']:.3f}\n")

    print(f"Testing complete. Results saved to {test_log}")
    return avg


# ---------- Main ----------
def main():

    # load datasets
    train_ds = MSADataset("train")
    val_ds = MSADataset("validation")
    test_ds = MSADataset("test")

    # train model
    agent = train_multitask(train_ds, val_ds)

    # load best checkpoint (best CS) for final testing
    best_ckpt = os.path.join(config.weight_path, "shared_dqn_final.pt")
    if os.path.exists(best_ckpt):
        agent.load(best_ckpt)
    else:
        print("Warning: best checkpoint not found, using last trained weights.")

    # test model
    test_model(agent, test_ds)


if __name__ == "__main__":
    main()