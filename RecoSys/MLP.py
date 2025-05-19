import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

num_classes = 5

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.linear4 = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


def weight_init(model):
  if isinstance(model, nn.Linear):
    torch.nn.init.kaiming_uniform_(model.weight.data)

model = MLP(num_classes).to(device)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

print(model)