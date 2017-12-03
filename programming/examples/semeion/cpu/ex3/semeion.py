from torch.utils.data import Dataset
import torch
import numpy as np
import urllib.request
import os

class SemeionDataset(Dataset):


    def __init__(self, root='./', train=None, transform=None):

        self.transform = transform

        path = root + 'semeion.data'

        # Download dataset if not available
        if os.path.isfile(path) is False:
            print("Downloading dataset...")
            self.__downloadDataset(path)
            print("Download finished...")

        # Load dataset
        self.__loadDataset(path)

        if train is True:
            self.semeion_dataset = self.semeion_dataset[0::2] # Every second element starting with the first
        elif train is False:
            self.semeion_dataset = self.semeion_dataset[1::2] # Every second element starting with the second


    def __len__(self):
        return len(self.semeion_dataset)


    def __getitem__(self, idx):

        data = self.semeion_dataset[idx].rstrip().split(" ")
        image_data = data[:-10]
        label_data = data[-10:]

        _1dImage = np.array(image_data, dtype=np.float64)
        _2dImage = np.reshape(_1dImage, (16, 16))

        label = self.__convertLabel(label_data)

        sample = (_2dImage, label)

        if self.transform:
            sample = self.transform(sample)

        return sample


    def __downloadDataset(self, path):
        urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data', path)


    def __loadDataset(self, path):
        self.f = open(path)
        self.semeion_dataset = self.f.readlines()
        self.f.close()


    def __convertLabel(self, labelAsList):

        for i, label in enumerate(labelAsList):
            if label == '1':
                return i
        return None



class ToTensor(object):

    def __call__(self, sample):

        data, label = sample

        data = data.reshape(1,16,16) # One channel

        imageTensor = torch.from_numpy(data).float()

        return (imageTensor,label)
