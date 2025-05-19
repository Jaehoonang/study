import os
import glob

from torch.utils.data import Dataset
from PIL import Image

class PlateDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = glob.glob(os.path.join(path, '*', '*.jpg'))
        self.transforms = transforms

        self.label_dict = {}
        sub_folder_names = os.listdir(path)
        for i, sub_folder_name in enumerate(sub_folder_names):
            self.label_dict[sub_folder_name] = i


    def __getitem__(self, item):
        image_path = self.path[item]
        image = Image.open(image_path).convert('RGB')

        sub_folder_name = image_path.split('\\')[1]
        label = self.label_dict[sub_folder_name]

        if self.transforms is not None:
            image = self.transforms(image)

        return  image, label

    def __len__(self):
        return  len(self.path)

if __name__ == '__main__':
    test = PlateDataset('./US_license_plates_dataset/train/', transforms=None)
    for i in test:
        pass
