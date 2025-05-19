import glob
import os
import matplotlib.pyplot as plt

class DataRatioView:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_data_dict = {}

    def load_data(self):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'valid')
        test_dir = os.path.join(self.data_dir, 'test')

        for train_sub_folder_name in os.listdir(train_dir):
            train_sub_folder_dir = os.path.join(train_dir, train_sub_folder_name)
            train_data_count = len(os.listdir(train_sub_folder_dir))
            self.all_data_dict[train_sub_folder_name] = train_data_count

        for val_sub_folder_name in os.listdir(val_dir):
            val_sub_folder_dir = os.path.join(val_dir, val_sub_folder_name)
            val_data_count = len(os.listdir(val_sub_folder_dir))
            if val_sub_folder_name in self.all_data_dict:
                self.all_data_dict[val_sub_folder_name] += val_data_count
            else:
                self.all_data_dict[val_sub_folder_name] = val_data_count

        for test_sub_folder_name in os.listdir(test_dir):
            test_sub_folder_dir = os.path.join(test_dir, test_sub_folder_name)
            test_data_count = len(os.listdir(test_sub_folder_dir))
            if test_sub_folder_name in self.all_data_dict:
                self.all_data_dict[test_sub_folder_name] += test_data_count
            else:
                self.all_data_dict[test_sub_folder_name] = test_data_count

    def visualization(self):
        labels = list(self.all_data_dict.keys())
        counts = list(self.all_data_dict.values())

        plt.figure(figsize=(10,6))
        plt.bar(labels, counts)
        plt.title('plate ratio by Region')
        plt.xlabel('Region')
        plt.ylabel('Freq')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.show()


if __name__ == '__main__':
    data_dir = './US_license_plates_dataset/'

    visualizer = DataRatioView(data_dir)
    visualizer.load_data()
    visualizer.visualization()