# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:41:15 2017

@author: yansel
"""

import numpy as np
from semeion import SEMEION

semeion_dataset = SEMEION('./data')

dataset_length = len(semeion_dataset)

# generate the indices so we can "shuffle" them
indices = list(range(dataset_length))

# the percentage of the whole dataset we are going to use as a test set
# (in this case the 20%)
test_set_size = 0.2

# get the index at with we have to split
split = int(np.floor(test_set_size * dataset_length))


np.random.seed(18)

np.random.shuffle(indices)

# now we have the index of the element in the dataset
training_idx, test_idx = indices[split:], indices[:split]
print('Use these indexes to pick elements to make a list to train your cnn')
print(training_idx)
print('and these to test how your cnn performs')
print(test_idx)