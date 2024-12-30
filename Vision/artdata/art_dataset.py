import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ArtDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = glob.glob(os.path.join(path, '*', '*.jpg'))
        self.transforms = transforms

        # label dictionary
        folders = os.listdir(path)
        self.label_dict = {}
        for i, sub_folder in enumerate(folders):
            self.label_dict[sub_folder] = i

    def __getitem__(self, item):
        image_path = self.path[item]

        image = Image.open(image_path).convert('RGB')


        folder_name = image_path.split('\\')[1]
        label_number = self.label_dict[folder_name]

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label_number

    def __len__(self):
        return len(self.path)

test = ArtDataset('./dataset/train/', transforms=None)

for i in test:
    pass
