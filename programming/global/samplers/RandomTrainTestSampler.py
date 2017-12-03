import torch
import random

class SubsetSampler(object):

    def __init__(self, subset):
        self.subset = subset

    def __iter__(self):
        return iter(self.subset)

    def __len__(self):
        return len(self.subset)


class RandomTrainTestSampler(object):

    def __init__(self, data_source, train_share=0.8):

        # Generate a list of indizes reaching from 0 ... len(data_source)-1
        idxList = list(range(0,len(data_source)))

        # Ensure that list is sorted randomly
        random.shuffle(idxList)

        # Split dataset random shares of train and test data
        numberOfTrainSamples = int(len(data_source) / (1 / train_share))
        
        self.train_samples = idxList[:numberOfTrainSamples]
        self.test_samples = idxList[numberOfTrainSamples:]

    def trainSampler(self):
        return SubsetSampler(self.train_samples)

    def testSampler(self):
        return SubsetSampler(self.test_samples)
