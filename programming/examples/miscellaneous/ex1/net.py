from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(28*28, 10)

    def forward(self, x):
        self.data = self.linear1(x)

        return F.softmax(self.data)

