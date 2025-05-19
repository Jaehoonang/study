import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large
from lion_pytorch import Lion
from custome_plate_dataset import PlateDataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
class Classifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = None
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def train(self, train_loader, val_loader, epochs, optimizer, criterion):
        best_val_acc = 0.0
        print('Train .....')

        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            val_loss = 0.0
            val_acc = 0.0

            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f'Epoch: [{epoch+1}/{epochs}]'), leave=False)

            for i, (images, labels) in enumerate(train_loader_iter):
                image = images.to(self.device)
                label = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(image)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                train_acc += (pred == label).sum().item()
                train_loader_iter.set_postfix({'loss':loss.item()})

            train_loss /= len(train_loader)
            train_acc = train_acc/len(train_loader.dataset)

            self.model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    image = images.to(self.device)
                    label = labels.to(self.device)

                    outputs = self.model(image)
                    pred = outputs.argmax(dim=1, keepdims=True)
                    val_acc += pred.eq(label.view_as(pred)).sum().item()
                    val_loss += criterion(outputs, label).item()

            val_loss /= len(val_loader)
            val_acc = val_acc / len(val_loader.dataset)

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)

            if val_acc > best_val_acc:
                torch.save(self.model.state_dict(), './best_model.pt')
                best_val_acc = val_acc

            print(f'Epoch [{epoch + 1}] / [{epochs}, Train Loss: {train_loss}, Val Loss : {val_loss}'
                  f' Train acc {train_acc}, Val acc {val_acc}')

        torch.save(self.model.state_dict(), './last_model.pt')

        self.save_result_to_csv()
        self.plot_loss_view()
        self.plot_acc_view()
    def save_result_to_csv(self):
        df = pd.DataFrame({
            'Train Loss': self.train_loss_list,
            'Val Loss': self.val_loss_list,
            'Train acc': self.train_acc_list,
            'Val acc': self.val_acc_list
        })
        df.to_csv('./train_val_result.csv', index=False)


    def plot_loss_view(self):
        plt.figure()
        plt.plot(self.train_loss_list, label='Train Loss')
        plt.plot(self.val_loss_list, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./loss_plot.png')

    def plot_acc_view(self):
        plt.figure()
        plt.plot(self.train_acc_list, label='Train acc')
        plt.plot(self.val_acc_list, label='Val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig('./accuracy_plot.png')
    def run(self):
        #augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=5),
            transforms.AugMix(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_dataset = PlateDataset('./US_license_plates_dataset/train/', transforms=train_transforms)
        val_dataset = PlateDataset('./US_license_plates_dataset/valid/', transforms=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        self.model = mobilenet_v3_large(weights=True)
        self.model.classifier[3] = nn.Linear(in_features=1280, out_features=50)
        self.model.to(device=self.device)

        epochs = 50
        criterion = CrossEntropyLoss().to(device=self.device)
        optimizer = Lion(self.model.parameters(), lr=0.001, weight_decay=1e-2)

        self.train(train_loader, val_loader, epochs, optimizer, criterion)

if __name__ == '__main__':
    classifier = Classifier()
    classifier.run()
