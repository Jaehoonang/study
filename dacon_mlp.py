import pandas as pd


from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

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
data['월세'] = scaler1.fit_transform(data['월세'].values.reshape(-1, 1))

jj = test_df.copy()
jj = jj.drop(columns=['ID', '게재일'])
jj.loc[(jj['주차가능여부'] == '불가능') & (jj['총주차대수'].isnull()), '총주차대수'] = 0

encoder = LabelEncoder()
jj['매물확인방식'] = encoder.fit_transform(jj['매물확인방식'])
jj['방향'] = encoder.fit_transform(jj['방향'])
jj['주차가능여부'] = encoder.fit_transform(jj['주차가능여부'])
jj['중개사무소'] = encoder.fit_transform(jj['중개사무소'])
jj['제공플랫폼'] = encoder.fit_transform(jj['제공플랫폼'])

imputer = KNNImputer(n_neighbors=7)
data_filled = imputer.fit_transform(jj)

cols = jj.columns
data1 = pd.DataFrame(data_filled, columns=cols)

data1['보증금'] = scaler1.fit_transform(data1['보증금'].values.reshape(-1, 1))

data1['월세'] = scaler1.fit_transform(data1['월세'].values.reshape(-1, 1))

y = data['허위매물여부']
x = data.drop(columns=['허위매물여부'])

dum_x = pd.get_dummies(x)
dum_test = pd.get_dummies(data1)

x_train_tensor = torch.tensor(dum_x.values, dtype=torch.float32)
x_test_tensor = torch.tensor(dum_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y.values)
y_train_tensor = y_train_tensor.squeeze()

dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, hidden_size1)
        self.fc6 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)

        return x

input_size = 14
hidden_size1 = 32
hidden_size2 = 64
hidden_size3 = 128
output_size = 2

model = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

model = model.to(device)

num_epochs = 5000

for epoch in tqdm(range(1, num_epochs + 1)):
    total_loss = 0.0
    total_accuracy = 0.0
    all_preds = []
    all_labels = []


    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

        accuracy = calculate_accuracy(outputs, batch_y)
        total_accuracy += accuracy

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    f1_macro = f1_score(all_labels, all_preds, average='macro')

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1-Macro: {f1_macro:.4f}")

model.eval()

with torch.no_grad():
  x_test_tensor = x_test_tensor.to(device)

  predictions = model(x_test_tensor)

_, predicted_labels = torch.max(predictions, 1)

predicted_labels = predicted_labels.cpu().numpy()

test_id = test_df['ID'].values.flatten()

result_df = pd.DataFrame({'ID': test_id, '허위매물여부': predicted_labels})
result_df.to_csv('submission_18.csv',index=False)
result_df.head()