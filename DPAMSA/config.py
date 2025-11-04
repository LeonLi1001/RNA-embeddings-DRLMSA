import os.path
import platform
import torch
import math

# ---------- Embedding vector max length----------
MAX_EMBED_VEC_LEN = 150

# ---------- Alignment scoring ----------
GAP_PENALTY = -4
MISMATCH_PENALTY = -4
MATCH_REWARD = 4

# ---------- Training hyperparameters (1.5M training, 2k for validation, 3k for testing) ----------
'''update_iteration = 128
batch_size = 128
max_episode = 300000
replay_memory_size = 200000'''
### for testing
update_iteration = 16
batch_size = 8
max_episode = 200             # only 200 episodes for quick test
replay_memory_size = 100

# ---------- Optimization ----------
alpha = 1e-4                 # learning rate
gamma = 0.99                 # discount factor (stability)
epsilon = 0.9                # initial epsilon for ε-greedy
delta = 0.05                 # epsilon decay step
epsilon_min = 0.05           # floor for exploration
epsilon_decay_ratio = 0.8    # fraction of training where epsilon decays

decrement_iteration = math.ceil(max_episode * 0.8 / (epsilon // delta))

# ---------- Evaluation ----------
'''eval_every = 5000           # validate every N episodes
save_every = 10000          # checkpoint every N episodes
val_samples = 500            # how many validation MSAs per eval
test_samples = 2500          # test set size'''
eval_every = 50
save_every = 100
val_samples = 3
test_samples = 3

# ---------- Device auto-selection ----------
if torch.cuda.is_available():
    device_name = "cuda:0"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

device = torch.device(device_name)
print(f"[CONFIG] Using device: {device_name}")

# ---------- Paths ----------
weight_path = "../result/weight"
score_path = "../result/score"
report_path = "../result/report"

if not os.path.exists(score_path):
    os.makedirs(score_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)
if not os.path.exists(report_path):
    os.makedirs(report_path)

# ---------- Assertions ----------
assert 0 < batch_size <= replay_memory_size, "batch size must be in the range of 0 to the size of replay memory."
assert alpha > 0, "alpha must be greater than 0."
assert 0 <= gamma <= 1, "gamma must be in the range of 0 to 1."
assert 0 <= epsilon <= 1, "epsilon must be in the range of 0 to 1."
assert 0 <= delta <= epsilon, "delta must be in the range of 0 to epsilon."
assert 0 < decrement_iteration, "decrement iteration must be greater than 0."


# ---------- Original Optimization ----------
'''alpha = 0.0001
gamma = 1
epsilon = 0.8
delta = 0.05'''

