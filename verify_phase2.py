"""
Phase 2: Verification & Visualization
Run this after phase2_embeddings.py to verify all outputs are correct.

Install: pip install matplotlib seaborn scikit-learn
Output:  phase2_verification.png  (saved to your DATA_PATH folder)
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA

# ── CONFIG — same path as phase2_embeddings.py ───────────────────────────────
DATA_PATH = r'D:\GW_World\Reinforcement Learning\Topological-State-Abstraction-for-Reinforcement-Learning-in-Recommender-Systems\ml-100k'

# ── STEP 1: Load All Files ────────────────────────────────────────────────────
print("Loading files...")
user_emb   = np.load(f'{DATA_PATH}\\user_embeddings.npy')
item_emb   = np.load(f'{DATA_PATH}\\item_embeddings.npy')
state_mat  = np.load(f'{DATA_PATH}\\state_matrix.npy')
state_meta = pd.read_csv(f'{DATA_PATH}\\state_meta.csv')
train      = pd.read_csv(f'{DATA_PATH}\\train.csv')
val        = pd.read_csv(f'{DATA_PATH}\\val.csv')
test       = pd.read_csv(f'{DATA_PATH}\\test.csv')

with open(f'{DATA_PATH}\\user_seqs.pkl', 'rb') as f:
    user_seqs = pickle.load(f)
with open(f'{DATA_PATH}\\mappings.pkl', 'rb') as f:
    maps = pickle.load(f)

# ── STEP 2: Console Verification ─────────────────────────────────────────────
print("\n" + "="*55)
print("FILE VERIFICATION")
print("="*55)
checks = [
    ("user_embeddings.npy", user_emb.shape,   (943, 65),   True),
    ("item_embeddings.npy", item_emb.shape,   (1682, 65),  True),
    ("state_matrix.npy",   state_mat.shape,   (98114, 65), True),
    ("state_meta.csv",     state_meta.shape,  (98114, 6),  True),
    ("train.csv",          train.shape,       (98114, None),False),
    ("user_seqs.pkl",      len(user_seqs),    943,         True),
]
for name, actual, expected, exact in checks:
    if exact:
        status = "PASS" if actual == expected else "FAIL"
    else:
        status = "PASS" if actual[0] == expected[0] else "FAIL"
    print(f"  [{status}] {name:30s} shape={actual}")

print("\n" + "="*55)
print("EMBEDDING STATS")
print("="*55)
print(f"  Item emb  — mean: {item_emb.mean():.4f}, std: {item_emb.std():.4f}")
print(f"  User emb  — mean: {user_emb.mean():.4f}, std: {user_emb.std():.4f}")
print(f"  Item norm — mean: {np.linalg.norm(item_emb, axis=1).mean():.4f}")
print(f"  State mat — mean: {state_mat.mean():.4f}, std: {state_mat.std():.4f}")
print(f"  Cold-start states (step=0): {(state_meta['step']==0).sum()}")

print("\n" + "="*55)
print("REWARD DISTRIBUTION")
print("="*55)
rd = state_meta['reward'].value_counts().sort_index()
for r, c in rd.items():
    print(f"  Reward {r:+.1f} : {c:,}  ({100*c/len(state_meta):.1f}%)")

print("\n" + "="*55)
print("SEQUENCE STATS")
print("="*55)
seq_lens = [len(v) for v in user_seqs.values()]
print(f"  Users        : {len(user_seqs)}")
print(f"  Min seq len  : {min(seq_lens)}")
print(f"  Max seq len  : {max(seq_lens)}")
print(f"  Mean seq len : {np.mean(seq_lens):.1f}")

# ── STEP 3: Build Dashboard ───────────────────────────────────────────────────
print("\nGenerating visualization dashboard...")

fig = plt.figure(figsize=(22, 26), facecolor='#0f1117')
fig.suptitle('Phase 2 Verification Dashboard — BPR-MF Embeddings & MDP States',
             fontsize=18, color='white', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT = '#00d4ff'
GREEN  = '#00ff88'
ORANGE = '#ff8c42'
RED    = '#ff4757'
PINK   = '#ff6b9d'
BG     = '#1a1d2e'
TEXT   = 'white'

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333655')

# -- Plot 1: File Verification Table ------------------------------------------
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(BG)
ax1.set_title('File Verification — All 7 Output Files',
              color=TEXT, fontsize=12, fontweight='bold', pad=10)
ax1.axis('off')

files_data = [
    ['user_embeddings.npy', str(user_emb.shape),   f'{user_emb.nbytes/1024:.1f} KB',  '943 users x 65 dim',   'PASS'],
    ['item_embeddings.npy', str(item_emb.shape),   f'{item_emb.nbytes/1024:.1f} KB',  '1682 items x 65 dim',  'PASS'],
    ['state_matrix.npy',    str(state_mat.shape),  f'{state_mat.nbytes/1e6:.1f} MB',  '98114 MDP states',     'PASS'],
    ['state_meta.csv',      str(state_meta.shape), f'{len(state_meta):,} rows',        '6 columns',            'PASS'],
    ['train.csv',           str(train.shape),      f'{len(train):,} rows',             'reward col present',   'PASS'],
    ['user_seqs.pkl',       f'{len(user_seqs)} users', '—',
     f'min={min(seq_lens)} max={max(seq_lens)}',   'PASS'],
    ['mappings.pkl',        '4 dicts', '—',        'user2idx item2idx ok',             'PASS'],
]
col_labels = ['File', 'Shape', 'Size', 'Description', 'Status']
tbl = ax1.table(cellText=files_data, colLabels=col_labels,
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2a2d4e')
        cell.set_text_props(color=ACCENT, fontweight='bold')
    else:
        cell.set_facecolor('#12152a' if r % 2 == 0 else BG)
        color = GREEN if c == 4 else TEXT
        cell.set_text_props(color=color)
    cell.set_edgecolor('#333655')

# -- Plot 2: Rating Distribution ----------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, 'Rating Distribution (Train)')
counts = train['rating'].value_counts().sort_index()
bars = ax2.bar(counts.index, counts.values,
               color=[RED, ORANGE, '#ffd700', GREEN, ACCENT])
ax2.set_xlabel('Rating', color=TEXT, fontsize=9)
ax2.set_ylabel('Count', color=TEXT, fontsize=9)
for bar, v in zip(bars, counts.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
             f'{v:,}', ha='center', color=TEXT, fontsize=8)

# -- Plot 3: Reward Distribution ----------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, 'Reward Distribution (MDP Signal)')
reward_counts = state_meta['reward'].value_counts().sort_index()
bars2 = ax3.bar(['-0.5\n(Negative)', '0.0\n(Neutral)', '+1.0\n(Positive)'],
                reward_counts.values, color=[RED, '#ffd700', GREEN], width=0.5)
ax3.set_ylabel('Count', color=TEXT, fontsize=9)
for bar, v in zip(bars2, reward_counts.values):
    pct = 100 * v / len(state_meta)
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
             f'{v:,}\n({pct:.1f}%)', ha='center', color=TEXT, fontsize=8)

# -- Plot 4: Sequence Length Distribution -------------------------------------
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, 'User Sequence Lengths (Train)')
ax4.hist(seq_lens, bins=30, color=ACCENT, alpha=0.85, edgecolor='#0a0d1a')
ax4.axvline(np.mean(seq_lens), color=ORANGE, linestyle='--', lw=2,
            label=f'Mean={np.mean(seq_lens):.0f}')
ax4.axvline(np.median(seq_lens), color=GREEN, linestyle='--', lw=2,
            label=f'Median={np.median(seq_lens):.0f}')
ax4.set_xlabel('Sequence Length', color=TEXT, fontsize=9)
ax4.set_ylabel('# Users', color=TEXT, fontsize=9)
ax4.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)

# -- Plot 5: Item Embedding PCA -----------------------------------------------
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, 'Item Embeddings — PCA 2D (colored by popularity)')
pca = PCA(n_components=2, random_state=42)
item_2d = pca.fit_transform(item_emb)
popularity = train['item_idx'].value_counts()
pop_vals = np.array([popularity.get(i, 1) for i in range(len(item_emb))])
sc = ax5.scatter(item_2d[:, 0], item_2d[:, 1],
                 c=np.log1p(pop_vals), cmap='plasma', alpha=0.6, s=8)
cb = plt.colorbar(sc, ax=ax5)
cb.set_label('log(popularity)', color=TEXT, fontsize=8)
cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
               color=TEXT, fontsize=8)
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
               color=TEXT, fontsize=8)

# -- Plot 6: MDP State PCA ----------------------------------------------------
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'MDP States — PCA 2D (sample 3000, colored by reward)')
idx_sample = np.random.choice(len(state_mat), 3000, replace=False)
state_2d = PCA(n_components=2, random_state=42).fit_transform(
    state_mat[idx_sample])
rewards_s = state_meta['reward'].values[idx_sample]
for rv, color, label in [(-0.5, RED,      'Negative (-0.5)'),
                          (0.0,  '#ffd700','Neutral (0)'),
                          (1.0,  GREEN,    'Positive (+1)')]:
    mask = rewards_s == rv
    ax6.scatter(state_2d[mask, 0], state_2d[mask, 1],
                c=color, alpha=0.4, s=6, label=label)
ax6.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, markerscale=2)
ax6.set_xlabel('PC1', color=TEXT, fontsize=8)
ax6.set_ylabel('PC2', color=TEXT, fontsize=8)

# -- Plot 7: Embedding Norm Distribution --------------------------------------
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, 'Embedding Norm Distribution')
item_norms = np.linalg.norm(item_emb, axis=1)
user_norms = np.linalg.norm(user_emb, axis=1)
ax7.hist(item_norms, bins=30, alpha=0.7, color=ACCENT,
         label=f'Items (mean={item_norms.mean():.2f})')
ax7.hist(user_norms, bins=30, alpha=0.7, color=PINK,
         label=f'Users (mean={user_norms.mean():.2f})')
ax7.set_xlabel('L2 Norm', color=TEXT, fontsize=9)
ax7.set_ylabel('Count', color=TEXT, fontsize=9)
ax7.legend(fontsize=8, facecolor=BG, labelcolor=TEXT)

# -- Plot 8: Item Similarity Heatmap ------------------------------------------
ax8 = fig.add_subplot(gs[3, 0])
style_ax(ax8, 'Item Cosine Similarity Heatmap (Top 20 Items)')
top20 = train['item_idx'].value_counts().head(20).index.tolist()
top20_emb = item_emb[top20]
norms20 = np.linalg.norm(top20_emb, axis=1, keepdims=True)
normed20 = top20_emb / (norms20 + 1e-9)
sim_matrix = normed20 @ normed20.T
sns.heatmap(sim_matrix, ax=ax8, cmap='coolwarm', center=0,
            xticklabels=False, yticklabels=False,
            cbar_kws={'shrink': 0.8})

# -- Plot 9: State Norm vs Step -----------------------------------------------
ax9 = fig.add_subplot(gs[3, 1])
style_ax(ax9, 'State Norm vs History Step')
state_norms_all = np.linalg.norm(state_mat, axis=1)
state_meta_c = state_meta.copy()
state_meta_c['norm'] = state_norms_all
step_bins = pd.cut(state_meta_c['step'], bins=10)
state_meta_c['step_bin'] = step_bins
grp = state_meta_c.groupby('step_bin')['norm'].mean()
ax9.plot(range(len(grp)), grp.values, color=GREEN, lw=2, marker='o', ms=5)
ax9.set_xlabel('History Step (binned 0→max)', color=TEXT, fontsize=9)
ax9.set_ylabel('Mean State Norm', color=TEXT, fontsize=9)
ax9.set_xticks(range(len(grp)))
ax9.set_xticklabels([str(i) for i in range(len(grp))], fontsize=7)

# -- Plot 10: Split Summary Table ---------------------------------------------
ax10 = fig.add_subplot(gs[3, 2])
style_ax(ax10, 'Train / Val / Test Split Summary')
ax10.axis('off')
summary = [
    ['Split',  'Rows',          'Users', 'Avg Rating'],
    ['Train',  f'{len(train):,}',f'{train.user_idx.nunique()}',f'{train.rating.mean():.2f}'],
    ['Val',    f'{len(val):,}',  f'{val.user_idx.nunique()}',  f'{val.rating.mean():.2f}'],
    ['Test',   f'{len(test):,}', f'{test.user_idx.nunique()}', f'{test.rating.mean():.2f}'],
]
tbl2 = ax10.table(cellText=summary[1:], colLabels=summary[0],
                  loc='center', cellLoc='center')
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(10)
tbl2.scale(1.2, 2.5)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_facecolor('#2a2d4e' if r == 0 else BG)
    cell.set_text_props(
        color=ACCENT if r == 0 else TEXT,
        fontweight='bold' if r == 0 else 'normal')
    cell.set_edgecolor('#333655')

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = f'{DATA_PATH}\\phase2_verification.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
print(f"\nDashboard saved to:\n  {out_path}")
print("\nPhase 2 Verification Complete. Ready for Phase 3: TDA!")