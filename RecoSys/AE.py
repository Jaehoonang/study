import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 40
EPOCHS = 10

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data = None):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.y_data is None:
            x = torch.Tensor(self.x_data[idx])
            return x
        else:
            x = torch.Tensor(self.x_data[idx])
            y = torch.Tensor(self.y_data[idx])
            return x, y

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2830, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,32)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.Linear(512, 2830)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(model, train_loader):
    model.train()
    train_loss = 0

    for batch_idx, (feature) in enumerate(train_loader):

        feature = feature.to(device)
        target = feature.to(device)
        optimizer.zero_grad()
        encoded, decoded = model(feature)
        loss = criterion(decoded. target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss

model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(model)