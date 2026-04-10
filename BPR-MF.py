"""
Phase 2: Embedding Generation — BPR-MF + MDP State Construction
RecSys ACM Project: Topological State Abstraction for RL-based Recommendation

Outputs:
  user_embeddings.npy  → (943,  65)  user latent factors
  item_embeddings.npy  → (1682, 65)  item latent factors
  state_matrix.npy     → (98114, 65) one MDP state per training transition
  state_meta.csv       → metadata (user, item, reward, step) per state row
  train/val/test.csv   → splits with reward column added
  user_seqs.pkl        → {user_idx: [item_idx, ...]} ordered sequences
  mappings.pkl         → user2idx, item2idx, idx2user, idx2item
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
import pickle

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH  = r'D:\GW_World\Reinforcement Learning\Topological-State-Abstraction-for-Reinforcement-Learning-in-Recommender-Systems\ml-100k'
OUT_PATH   = DATA_PATH          # save outputs to same folder
EMBED_DIM  = 64                 # BPR-MF latent dimension
N          = 10                 # MDP history window (ablate: 5, 10, 20)
BPR_ITERS  = 100                # BPR training iterations
BPR_LR     = 0.01              # learning rate
BPR_REG    = 0.01              # L2 regularization

# ── STEP 1: Load & Re-index ──────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Load & Re-index")
print("=" * 55)

df = pd.read_csv(f'{DATA_PATH}\\u.data', sep='\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# 0-based contiguous index maps
user2idx = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
item2idx = {v: i for i, v in enumerate(sorted(df.item_id.unique()))}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: v for v, i in item2idx.items()}

df['user_idx'] = df['user_id'].map(user2idx)
df['item_idx'] = df['item_id'].map(item2idx)

NUM_USERS = len(user2idx)    # 943
NUM_ITEMS = len(item2idx)    # 1682
print(f"  Users : {NUM_USERS} | Items : {NUM_ITEMS}")

# ── STEP 2: Train / Val / Test Split (Leave-Last-Out) ───────────────────────
print("\n" + "=" * 55)
print("STEP 2: Leave-Last-Out Split")
print("=" * 55)

train_list, val_list, test_list = [], [], []
for uid, grp in df.groupby('user_idx'):
    rows = grp.reset_index(drop=True)
    n = len(rows)
    if n >= 3:
        train_list.append(rows.iloc[:-2])
        val_list.append(rows.iloc[[-2]])
        test_list.append(rows.iloc[[-1]])
    elif n == 2:
        train_list.append(rows.iloc[:1])
        val_list.append(rows.iloc[[1]])
    else:
        train_list.append(rows)

train = pd.concat(train_list).reset_index(drop=True)
val   = pd.concat(val_list).reset_index(drop=True)
test  = pd.concat(test_list).reset_index(drop=True)
print(f"  Train : {len(train):,} | Val : {len(val)} | Test : {len(test)}")

# ── STEP 3: Add Reward Column ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Binarize Ratings → Reward")
print("=" * 55)

def rating_to_reward(r):
    """MDP reward function — ablate threshold (3 vs 4)."""
    if r >= 4:  return  1.0    # positive engagement
    if r <= 2:  return -0.5    # negative engagement
    return 0.0                 # neutral

for split in [train, val, test]:
    split['reward'] = split['rating'].apply(rating_to_reward)

print(f"  Train reward dist: {train.reward.value_counts().sort_index().to_dict()}")

# ── STEP 4: Sparse Interaction Matrix ───────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Build Sparse Interaction Matrix")
print("=" * 55)

rows_idx = train['user_idx'].values
cols_idx = train['item_idx'].values
data     = np.ones(len(train), dtype=np.float32)

interaction_matrix = csr_matrix(
    (data, (rows_idx, cols_idx)),
    shape=(NUM_USERS, NUM_ITEMS)
)
sparsity = 1 - interaction_matrix.nnz / (NUM_USERS * NUM_ITEMS)
print(f"  Shape    : {interaction_matrix.shape}")
print(f"  Non-zero : {interaction_matrix.nnz:,}")
print(f"  Sparsity : {sparsity:.4f}")

# ── STEP 5: Train BPR-MF ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Train BPR-MF")
print("=" * 55)

model = implicit.bpr.BayesianPersonalizedRanking(
    factors        = EMBED_DIM,
    learning_rate  = BPR_LR,
    regularization = BPR_REG,
    iterations     = BPR_ITERS,
    random_state   = 42,
    verify_negative_samples = True
)
model.fit(interaction_matrix)

user_embeddings = model.user_factors   # (943,  65) — 64 factors + 1 bias
item_embeddings = model.item_factors   # (1682, 65)

print(f"\n  User embeddings : {user_embeddings.shape}")
print(f"  Item embeddings : {item_embeddings.shape}")
print(f"  Embedding norm  : {np.linalg.norm(item_embeddings, axis=1).mean():.4f}")

# ── STEP 6: Build User Sequences ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Build User Sequences")
print("=" * 55)

user_seqs = {}
for uid, grp in train.groupby('user_idx'):
    user_seqs[uid] = grp['item_idx'].tolist()   # chronological order

print(f"  Sequences built : {len(user_seqs)} users")
seq_lens = [len(v) for v in user_seqs.values()]
print(f"  Seq length — Min: {min(seq_lens)}, Max: {max(seq_lens)}, "
      f"Mean: {np.mean(seq_lens):.1f}")

# ── STEP 7: Build MDP State Matrix ──────────────────────────────────────────
print("\n" + "=" * 55)
print(f"STEP 7: Build MDP States (window N={N})")
print("=" * 55)

def build_state(history_items, item_emb, N):
    """
    MDP state = mean-pool of last N item embeddings.
    Returns zero vector for cold-start (no history yet).
    """
    if len(history_items) == 0:
        return np.zeros(item_emb.shape[1], dtype=np.float32)
    window = history_items[-N:]
    return item_emb[window].mean(axis=0).astype(np.float32)

state_rows, state_meta_list = [], []
for uid, grp in train.groupby('user_idx'):
    items_so_far = []
    for _, row in grp.iterrows():
        s = build_state(items_so_far, item_embeddings, N)
        state_rows.append(s)
        state_meta_list.append({
            'user_idx'  : int(uid),
            'item_idx'  : int(row['item_idx']),
            'rating'    : int(row['rating']),
            'reward'    : float(row['reward']),
            'timestamp' : int(row['timestamp']),
            'step'      : len(items_so_far)          # position in user history
        })
        items_so_far.append(int(row['item_idx']))

state_matrix   = np.vstack(state_rows).astype(np.float32)
state_meta_df  = pd.DataFrame(state_meta_list)

print(f"  State matrix shape : {state_matrix.shape}")
print(f"  Cold-start states  : {(state_matrix.sum(axis=1)==0).sum()} "
      f"(one per user — first interaction)")
print(f"  State mean         : {state_matrix.mean():.4f}")
print(f"  State std          : {state_matrix.std():.4f}")

# ── STEP 8: Save All Outputs ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 8: Save Outputs")
print("=" * 55)

np.save(f'{OUT_PATH}\\user_embeddings.npy', user_embeddings)
np.save(f'{OUT_PATH}\\item_embeddings.npy', item_embeddings)
np.save(f'{OUT_PATH}\\state_matrix.npy',   state_matrix)
state_meta_df.to_csv(f'{OUT_PATH}\\state_meta.csv',   index=False)
train.to_csv(f'{OUT_PATH}\\train.csv', index=False)
val.to_csv(f'{OUT_PATH}\\val.csv',     index=False)
test.to_csv(f'{OUT_PATH}\\test.csv',   index=False)

with open(f'{OUT_PATH}\\user_seqs.pkl', 'wb') as f:
    pickle.dump(user_seqs, f)
with open(f'{OUT_PATH}\\mappings.pkl', 'wb') as f:
    pickle.dump({'user2idx': user2idx, 'item2idx': item2idx,
                 'idx2user': idx2user, 'idx2item': idx2item,
                 'NUM_USERS': NUM_USERS, 'NUM_ITEMS': NUM_ITEMS}, f)

print(f"\n  ✅ user_embeddings.npy  → {user_embeddings.shape}")
print(f"  ✅ item_embeddings.npy  → {item_embeddings.shape}")
print(f"  ✅ state_matrix.npy     → {state_matrix.shape}")
print(f"  ✅ state_meta.csv       → {state_meta_df.shape}")
print(f"  ✅ train/val/test.csv")
print(f"  ✅ user_seqs.pkl        → {len(user_seqs)} users")
print(f"  ✅ mappings.pkl         → user2idx, item2idx")
print(f"\nPhase 2 Complete. Ready for Phase 3: TDA State Abstraction.")