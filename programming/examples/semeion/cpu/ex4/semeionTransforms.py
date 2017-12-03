import numpy as np
import torch

# Converts the 1-channel image into a 3-channel mage
class ExpandTo3D(object):

    def __call__(self, image):

        width = image.size()[1]
        height = image.size()[2]

        return image.expand(3,width,height)


# Transform 10-dimensional label data into 1-dimension
# Example: 0 0 1 0 0 0 0 0 0 0 => 2
class TransformLabel(object):

    def __call__(self, label):

        idx = np.where(label==1)[0][0]

        return idx
