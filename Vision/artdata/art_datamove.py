import os
import random
import shutil

# 데이터 경로 설정
main_folder_path = './art_data/'
dataset_path = './dataset'

train_folder_path = os.path.join(dataset_path, 'train')
val_folder_path = os.path.join(dataset_path, 'val')

os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)

# sub folder 생성

sub_folders = os.listdir(main_folder_path)

for sub_folder in sub_folders:
    sub_folder_full_path = os.path.join(main_folder_path, sub_folder)
    images = os.listdir(sub_folder_full_path)
    random.shuffle(images)

    train_sub_folder_path = os.path.join(train_folder_path, sub_folder)
    val_sub_folder_path = os.path.join(val_folder_path, sub_folder)

    os.makedirs(train_sub_folder_path, exist_ok=True)
    os.makedirs(val_sub_folder_path, exist_ok=True)

    # train-validation split(9:1)
    split_index = int(len(images) * 0.9)

    # train image move
    for image in images[:split_index]:
        src = os.path.join(sub_folder_full_path, image)
        dst = os.path.join(train_sub_folder_path, image)
        shutil.copyfile(src,dst)

    for image in images[split_index:]:
        src = os.path.join(sub_folder_full_path, image)
        dst = os.path.join(val_folder_path, image)
        shutil.copyfile(src, dst)

print('image move is done!')
