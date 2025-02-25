import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

test_df = pd.read_csv("C:/Users/12wkd/Desktop/recsys_paper_review/open1/test.csv")
train_df = pd.read_csv("C:/Users/12wkd/Desktop/recsys_paper_review/open1/train.csv")

kk = train_df.copy()
kk = kk.drop(columns=['ID', '게재일'])
kk.loc[(kk['주차가능여부'] == '불가능') & (kk['총주차대수'].isnull()), '총주차대수'] = 0

encoder = LabelEncoder()
kk['매물확인방식'] = encoder.fit_transform(kk['매물확인방식'])
kk['방향'] = encoder.fit_transform(kk['방향'])
kk['주차가능여부'] = encoder.fit_transform(kk['주차가능여부'])
kk['중개사무소'] = encoder.fit_transform(kk['중개사무소'])
kk['제공플랫폼'] = encoder.fit_transform(kk['제공플랫폼'])

imputer = KNNImputer(n_neighbors=7)
data_filled = imputer.fit_transform(kk)

cols = kk.columns
data = pd.DataFrame(data_filled, columns=cols)

scaler1 = MinMaxScaler()
data['보증금'] = scaler1.fit_transform(data['보증금'].values.reshape(-1, 1))

scaler2 = MinMaxScaler()
data['월세'] = scaler2.fit_transform(data['월세'].values.reshape(-1, 1))

jj = test_df.copy()
jj = jj.drop(columns=['ID', '게재일'])
jj.loc[(jj['주차가능여부'] == '불가능') & (jj['총주차대수'].isnull()), '총주차대수'] = 0

encoder = LabelEncoder()
jj['매물확인방식'] = encoder.fit_transform(jj['매물확인방식'])
jj['방향'] = encoder.fit_transform(jj['방향'])
jj['주차가능여부'] = encoder.fit_transform(jj['주차가능여부'])
jj['중개사무소'] = encoder.fit_transform(jj['중개사무소'])
jj['제공플랫폼'] = encoder.fit_transform(jj['제공플랫폼'])

imputer = KNNImputer(n_neighbors=4)
data_filled = imputer.fit_transform(jj)

cols = jj.columns
data1 = pd.DataFrame(data_filled, columns=cols)

data1['보증금'] = scaler1.fit_transform(data1['보증금'].values.reshape(-1, 1))

data1['월세'] = scaler2.fit_transform(data1['월세'].values.reshape(-1, 1))

y = data['허위매물여부']
x = data.drop(columns=['허위매물여부'])

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.11, random_state=42, stratify=y
)

# One-hot encoding for categorical features (after splitting)
dum_x_train = pd.get_dummies(x_train)
dum_x_val = pd.get_dummies(x_val)
dum_test = pd.get_dummies(data1)

# Ensure column alignment (important if train/test columns differ)
dum_x_val = dum_x_val.reindex(columns=dum_x_train.columns, fill_value=0)
dum_test = dum_test.reindex(columns=dum_x_train.columns, fill_value=0)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(dum_x_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(dum_x_val.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
x_test_tensor = torch.tensor(dum_test.values, dtype=torch.float32)

# Create datasets
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        # self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.35)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1 // 2)
        self.fc3 = nn.Linear(hidden_size1 // 2, output_size)
        # self.fc4 = nn.Linear(hidden_size3, hidden_size2)
        # self.fc5 = nn.Linear(hidden_size2, 16)
        # self.fc6 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc4(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc5(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc6(x)

        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

input_size = 14
hidden_size1 = 32
hidden_size2 = 128
hidden_size3 = 256
output_size = 2

model = MLP(input_size, hidden_size1, output_size)
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

def evaluate_model(model, val_dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(val_dataloader)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, f1_macro

model = model.to(device)

num_epochs = 2500

for epoch in tqdm(range(1, num_epochs + 1)):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    avg_train_loss = total_loss / len(train_dataloader)
    f1_macro_train = f1_score(all_labels, all_preds, average='macro')

    # Validation evaluation
    val_loss, val_f1_macro = evaluate_model(model, val_dataloader)

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train F1-Macro: {f1_macro_train:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1-Macro: {val_f1_macro:.4f}")

model.eval()

with torch.no_grad():
  x_test_tensor = x_test_tensor.to(device)

  predictions = model(x_test_tensor)

_, predicted_labels = torch.max(predictions, 1)

predicted_labels = predicted_labels.cpu().numpy()

test_id = test_df['ID'].values.flatten()

result_df = pd.DataFrame({'ID': test_id, '허위매물여부': predicted_labels})
print(result_df.head())

result_df.to_csv('C:/Users/12wkd/Desktop/recsys_paper_review/submission_20.csv',index=False)