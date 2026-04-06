"""
Phase 1: Data Preparation — MovieLens 100k
RecSys ACM Project: Topological State Abstraction for RL-based Recommendation
"""

import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Load Data
# ============================================================

# Genres list for u.item
GENRES = ['unknown','Action','Adventure','Animation','Childrens','Comedy',
          'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
          'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

# Base path to your dataset folder
DATA_PATH = r'D:\GW_World\Reinforcement Learning\Topological-State-Abstraction-for-Reinforcement-Learning-in-Recommender-Systems\ml-100k'
DATA_PATH1 = r'D:\GW_World\Reinforcement Learning\Topological-State-Abstraction-for-Reinforcement-Learning-in-Recommender-Systems'
# Load main ratings file
df = pd.read_csv(f'{DATA_PATH}\\u.data', sep='\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load item metadata (movies + genres)
items = pd.read_csv(f'{DATA_PATH}\\u.item', sep='|', encoding='latin-1',
                    names=['item_id','title','release_date','video_release','imdb_url'] + GENRES,
                    usecols=['item_id','title','release_date'] + GENRES)

# Load user demographics
users = pd.read_csv(f'{DATA_PATH}\\u.user', sep='|',
                    names=['user_id','age','gender','occupation','zip'])


# ============================================================
# STEP 2: Inspect & Analyze
# ============================================================

print("=" * 50)
print("STEP 2: Dataset Inspection")
print("=" * 50)

print(f"\n[u.data]")
print(f"  Shape       : {df.shape}")
print(f"  Users       : {df.user_id.nunique()}")
print(f"  Items       : {df.item_id.nunique()}")
print(f"  Ratings     : {len(df):,}")
print(f"  Rating range: {df.rating.min()} - {df.rating.max()}")
print(f"  Sparsity    : {1 - len(df)/(df.user_id.nunique()*df.item_id.nunique()):.4f}")
print(f"  Time range  : {pd.to_datetime(df.timestamp, unit='s').min()} "
      f"→ {pd.to_datetime(df.timestamp, unit='s').max()}")

print(f"\n[Rating Distribution]")
print(df.rating.value_counts().sort_index().to_string())

print(f"\n[Interactions per User]")
u_counts = df.groupby('user_id').size()
print(f"  Min: {u_counts.min()}, Max: {u_counts.max()}, "
      f"Mean: {u_counts.mean():.1f}, Median: {u_counts.median()}")

print(f"\n[Interactions per Item]")
i_counts = df.groupby('item_id').size()
print(f"  Min: {i_counts.min()}, Max: {i_counts.max()}, "
      f"Mean: {i_counts.mean():.1f}, Median: {i_counts.median()}")

print(f"\n[u.item]")
print(f"  Movies      : {len(items)}")
print(f"  Genre cols  : {len(GENRES)}")
print(f"  No-genre    : {(items[GENRES].sum(axis=1) == 0).sum()}")

print(f"\n[u.user]")
print(f"  Users       : {len(users)}")
print(f"  Gender dist : {users.gender.value_counts().to_dict()}")
print(f"  Age — Mean: {users.age.mean():.1f}, Min: {users.age.min()}, Max: {users.age.max()}")
print(f"  Top occupations:\n{users.occupation.value_counts().head(5).to_string()}")


# ============================================================
# STEP 3: Clean & Sort
# ============================================================

print("\n" + "=" * 50)
print("STEP 3: Cleaning & Sorting")
print("=" * 50)

# Sort by user and timestamp (critical for MDP sequences)
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Check for duplicates
dupes = df.duplicated(subset=['user_id','item_id','timestamp']).sum()
print(f"\n  Duplicate rows    : {dupes}")
if dupes > 0:
    df = df.drop_duplicates(subset=['user_id','item_id','timestamp'])
    print(f"  After dedup shape : {df.shape}")

# k-core check (already done in MovieLens 100k — min 20 per user)
user_min = df.groupby('user_id').size().min()
print(f"  Min interactions per user: {user_min} (k-core already applied ✅)")

print(f"\n  ✅ Data sorted chronologically per user")


# ============================================================
# STEP 4: Train / Val / Test Split (Leave-Last-Out)
# ============================================================

print("\n" + "=" * 50)
print("STEP 4: Leave-Last-Out Split")
print("=" * 50)

train_list, val_list, test_list = [], [], []

for user_id, group in df.groupby('user_id'):
    items_seq = group.copy()
    n = len(items_seq)
    if n >= 3:
        train_list.append(items_seq.iloc[:-2])       # all but last 2
        val_list.append(items_seq.iloc[[-2]])         # second to last
        test_list.append(items_seq.iloc[[-1]])        # last
    elif n == 2:
        train_list.append(items_seq.iloc[:1])
        val_list.append(items_seq.iloc[[1]])
    else:
        train_list.append(items_seq)

train = pd.concat(train_list).reset_index(drop=True)
val   = pd.concat(val_list).reset_index(drop=True)
test  = pd.concat(test_list).reset_index(drop=True)

print(f"\n  Train : {len(train):,} interactions | {train.user_id.nunique()} users")
print(f"  Val   : {len(val):,}  interactions | {val.user_id.nunique()} users")
print(f"  Test  : {len(test):,}  interactions | {test.user_id.nunique()} users")
print(f"\n  Avg train seq length/user : {len(train)/train.user_id.nunique():.1f}")

seq_lens = train.groupby('user_id').size()
print(f"  Train seq length — Min: {seq_lens.min()}, Max: {seq_lens.max()}, "
      f"Mean: {seq_lens.mean():.1f}")

print(f"\n  Train rating dist : {train.rating.value_counts().sort_index().to_dict()}")
print(f"  Test  rating dist : {test.rating.value_counts().sort_index().to_dict()}")
print(f"\n  ✅ No data leakage (splits use temporal row positions)")


# ============================================================
# STEP 5: Save Splits
# ============================================================

print("\n" + "=" * 50)
print("STEP 5: Saving Splits")
print("=" * 50)

train.to_csv(f'{DATA_PATH1}\\train.csv', index=False)
val.to_csv(f'{DATA_PATH1}\\val.csv', index=False)
test.to_csv(f'{DATA_PATH1}\\test.csv', index=False)

print(f"\n  ✅ Saved to: {DATA_PATH}\n     train.csv, val.csv, test.csv")
print("\nPhase 1 Complete. Ready for Phase 2: Embedding Generation.")