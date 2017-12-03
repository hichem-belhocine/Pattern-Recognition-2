import numpy as np
import torch

class Flatten(object):

    def __init__(self):
        pass

    def __call__(self, sample):

        image = np.array(sample)
        image1d = np.ndarray.flatten(image)

        return image1d


class Normalize(object):

    def __init__(self, normalizationFactor):
        self.normalizationFactor = normalizationFactor

    def __call__(self, sample):

        return sample / self.normalizationFactor


class ToTensor(object):

    def __call__(self, sample):

        tensor = torch.from_numpy(sample)
        tensor = tensor.float()

        return tensor
