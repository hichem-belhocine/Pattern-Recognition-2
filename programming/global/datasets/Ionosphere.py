from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class Ionosphere(data.Dataset):

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    filename = "ionosphere.data"
    md5_checksum = '85649e5fb5b15fb9dab726c400be61fe'

    def __init__(self, root, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.data = []
        self.labels = []
        fp = os.path.join(root, self.filename)
        data = np.loadtxt(fp, dtype='str', delimiter=',')

        self.data = (data[:, :34]).astype('float32')
        self.labels = data[:, 34]
        
        self.labels = list(map(lambda x: 1 if x == 'g' else 0,self.labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sensor_data, target = self.data[index], self.labels[index]
        
        if self.transform is not None:
            sensor_data = self.transform(sensor_data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sensor_data, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.md5_checksum)
