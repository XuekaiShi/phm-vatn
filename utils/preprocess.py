import os
import shutil
import random

import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

DATA_RAW_PATH = 'JNUData'
DATA_PATH = 'data'

SAMPLES_NUMBER = 1000
SAMPLE_WINDOW = 1000

TEST_RATE = 0.2
TRAIN_SET_PATH = os.path.join(DATA_PATH, 'train')
TEST_SET_PATH = os.path.join(DATA_PATH, 'test')

# BATCH_SIZE = 128
LABELS_MAP = {'n': 0, 't': 1, 'o': 2, 'i': 3}

def preprocess_data():

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    raw_files_names = os.listdir(DATA_RAW_PATH)

    for file_name in raw_files_names:
        file_path = os.path.join(DATA_RAW_PATH, file_name)
        print(file_path)
        data = pd.read_csv(file_path)
        for i in range(SAMPLES_NUMBER):
            # print(int(i/2.0*SAMPLE_WINDOW), int((i/2.0+1)*SAMPLE_WINDOW))
            sample = data.iloc[int(i/2.0*SAMPLE_WINDOW)
                                    :int((i/2.0+1)*SAMPLE_WINDOW-1)]
            new_file_name = file_name.split('_')[0]
            # print(new_file_name + '_' + str(i))
            sample.to_csv(os.path.join(
                DATA_PATH, f'{new_file_name}_{i}.csv'), index=False)

    files_names = os.listdir(DATA_PATH)

    for file_name in files_names:
        file_path = os.path.join(DATA_PATH, file_name)
        class_folder = file_name[0]
        if not os.path.exists(os.path.join(DATA_PATH, class_folder)):
            os.mkdir(os.path.join(DATA_PATH, class_folder))
        shutil.move(file_path, os.path.join(
            DATA_PATH, class_folder, file_name))

    classes_folder = os.listdir(DATA_PATH)

    if not os.path.exists(TRAIN_SET_PATH):
        os.mkdir(TRAIN_SET_PATH)
    if not os.path.exists(TEST_SET_PATH):
        os.mkdir(TEST_SET_PATH)

    for class_folder in classes_folder:
        files_names = os.listdir(os.path.join(DATA_PATH, class_folder))
        random.shuffle(files_names)
        testset_number = int(len(files_names) * TEST_RATE)
        testset_files_names = files_names[:testset_number]
        trainset_files_names = files_names[testset_number:]

        if not os.path.exists(os.path.join(TRAIN_SET_PATH, class_folder)):
            os.mkdir(os.path.join(TRAIN_SET_PATH, class_folder))
        if not os.path.exists(os.path.join(TEST_SET_PATH, class_folder)):
            os.mkdir(os.path.join(TEST_SET_PATH, class_folder))

        for testset_file_name in testset_files_names:
            file_path = os.path.join(
                DATA_PATH, class_folder, testset_file_name)
            shutil.move(file_path, os.path.join(
                TEST_SET_PATH, class_folder, testset_file_name))

        for trainset_file_name in trainset_files_names:
            file_path = os.path.join(
                DATA_PATH, class_folder, trainset_file_name)
            shutil.move(file_path, os.path.join(
                TRAIN_SET_PATH, class_folder, trainset_file_name))
        assert len(os.listdir(os.path.join(DATA_PATH, class_folder))) == 0
        shutil.rmtree(os.path.join(DATA_PATH, class_folder))

class JNUDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        self.data_path = data_path
        if is_train:
            path = os.path.join(data_path, 'train')
            classes_names = os.listdir(path)
        else:
            path = os.path.join(data_path, 'test')
            classes_names = os.listdir(path)
        self.data = []
        self.labels = []
        for class_name in classes_names:
            files_names = os.listdir(os.path.join(path, class_name))
            for file_name in files_names:
                tmp = pd.read_csv(os.path.join(path, class_name, file_name), header=None)
                tmp = tmp.values.squeeze().tolist()
                self.data.append(tmp)
                self.labels.append(LABELS_MAP[class_name])
        self.data = torch.tensor(self.data)
        self.data = torch.unsqueeze(self.data, 1)
        self.labels = torch.tensor(self.labels)
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label