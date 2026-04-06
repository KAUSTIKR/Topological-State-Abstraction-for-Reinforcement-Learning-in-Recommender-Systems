# BPR-MF.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# =====================
# 1️⃣ Dataset class
# =====================
class BPRDataset(Dataset):
    def __init__(self, interactions, num_items, neg_sample=1):
        """
        interactions: pd.DataFrame with columns ['user_id', 'item_id']
        num_items: total number of items
        """
        self.interactions = interactions
        self.user_item_set = set(zip(interactions['user_id'], interactions['item_id']))
        self.users = interactions['user_id'].unique()
        self.num_items = num_items
        self.neg_sample = neg_sample

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user = self.interactions.iloc[idx]['user_id']
        pos_item = self.interactions.iloc[idx]['item_id']

        # Negative sampling
        neg_item = np.random.randint(0, self.num_items)
        while (user, neg_item) in self.user_item_set:
            neg_item = np.random.randint(0, self.num_items)

        return torch.tensor(user, dtype=torch.long), \
               torch.tensor(pos_item, dtype=torch.long), \
               torch.tensor(neg_item, dtype=torch.long)

# =====================
# 2️⃣ BPR-MF Model
# =====================
class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(BPRMF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, pos_item, neg_item):
        u = self.user_emb(user)
        i = self.item_emb(pos_item)
        j = self.item_emb(neg_item)

        pos_score = torch.sum(u * i, dim=1)
        neg_score = torch.sum(u * j, dim=1)
        return pos_score, neg_score

# =====================
# 3️⃣ BPR Loss
# =====================
def bpr_loss(pos_score, neg_score):
    loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
    return loss

# =====================
# 4️⃣ Training function
# =====================
def train(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for user, pos, neg in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()
            pos_score, neg_score = model(user, pos, neg)
            loss = bpr_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(dataloader):.4f}")
    return model

# =====================
# 5️⃣ Main execution
# =====================
if __name__ == "__main__":
    # Load MovieLens 100K
    df = pd.read_csv('u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])
    
    # Only keep positive interactions (rating >= 4)
    df = df[df['rating'] >= 4].copy()

    # Shift IDs to zero-based indexing for PyTorch embeddings
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()

    dataset = BPRDataset(df, num_items)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Initialize model
    model = BPRMF(num_users, num_items, embedding_dim=64)

    # Train
    model = train(model, dataloader, epochs=10, lr=1e-3, device='cpu')

    # Save model
    torch.save(model.state_dict(), 'bpr_mf.pth')
    print("Model saved as bpr_mf.pth")