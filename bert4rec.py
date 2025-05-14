import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_movielens_100k(path="C:/Users/2wkd/Desktop/study/u.data"):
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user'] = user_encoder.fit_transform(df['user'])
    df['item'] = item_encoder.fit_transform(df['item'])
    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    user_seq = defaultdict(list)
    for row in df.itertuples():
        user_seq[row.user].append(row.item)

    sequences = list(user_seq.values())
    return sequences, item_num

class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, max_len=50, n_layers=2, n_heads=2, dropout=0.2):
        super(BERT4Rec, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout=dropout,activation='gelu',batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(hidden_dim, num_items + 2)

    def forward(self, input_seq):
        positions = torch.arange(input_seq.size(1), device=input_seq.device).unsqueeze(0).expand_as(input_seq)
        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        encoded = self.encoder(x)
        logits = self.linear(encoded)
        return logits

def mask_sequence(seq, mask_token_id, mask_prob=0.15):
    masked_seq = []
    labels = []
    for item in seq:
        if torch.rand(1).item() < mask_prob:
            masked_seq.append(mask_token_id)
            labels.append(item)
        else:
            masked_seq.append(item)
            labels.append(0)
    return masked_seq, labels

class SeqDataset(Dataset):
    def __init__(self, user_sequences, max_len, mask_token_id):
        self.sequences = user_sequences
        self.max_len = max_len
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][-self.max_len:]
        pad_len = self.max_len - len(seq)
        seq = [0] * pad_len + seq
        masked_seq, labels = mask_sequence(seq, self.mask_token_id)
        return torch.tensor(masked_seq), torch.tensor(labels)

def evaluate(model, test_tuples, item_num, mask_token_id, top_k=10):
    model.eval()
    hits, ndcgs = [], []
    with torch.no_grad():
        for seq, true_item in test_tuples:
            seq = seq[-50:]
            pad_len = 50 - len(seq)
            seq = [0]
            input_tensor = torch.tensor([seq], device=device)
            logits = model(input_tensor)
            scores = logits[0,-1]
            _, topk_items = torch.topk(scores, top_k)

            topk_items = topk_items.tolist()
            hit = int(true_item in topk_items)
            hits.append(hit)

            if hit:
                rank = topk_items.index(true_item) + 1
                ndcgs.append(1 / np.log2(rank + 1))
            else:
                ndcgs.append(0)
    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    print(f'HR@{top_k}: {hr:.4f}, NDCG@{top_k}: {ndcg:.4f}')

def train():
    max_len = 50
    batch_size = 128
    epochs = 100
    lr = 1e-4
    user_sequences, num_items = load_movielens_100k('u.data')  # 경로 확인 필요
    mask_token_id = num_items + 1

    train_seqs = []
    test_tuples = []

    for seq in user_sequences:
        if len(seq) <2:
            continue
        train_seqs.append(seq[:-1])
        test_tuples.append((seq[:-1], seq[-1]))

    model = BERT4Rec(num_items, max_len)
    model = model.to(device)
    optmizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_dataset = SeqDataset(user_sequences, max_len, mask_token_id)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for masked_seq ,labels in train_loader:
            masked_seq, labels = masked_seq.to(device), labels.to(device)
            logits = model(masked_seq)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}')

    evaluate(model, test_tuples, num_items, mask_token_id, top_k=10)

if __name__ == "__main__":
    train()