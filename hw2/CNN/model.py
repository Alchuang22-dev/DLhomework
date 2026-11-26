import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):

    def __init__(self, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        """TODO CNN: implement ResidualBlock.forward according to the handout"""
        raise NotImplementedError('TODO CNN: implement ResidualBlock.forward according to the handout')

class DistanceCNN(nn.Module):

    def __init__(self, mode='plain'):
        super().__init__()
        if mode not in ('plain', 'dilated'):
            raise ValueError(mode)
        self.stem = nn.Conv2d(20, 16, 1)
        self.block1 = ResidualBlock(1)
        self.block2 = ResidualBlock(2 if mode == 'dilated' else 1)
        self.head = nn.Conv2d(16, 16, 1)

    def raw(self, x):
        return self.head(self.block2(self.block1(F.relu(self.stem(x)))))

    def forward(self, x):
        """TODO CNN: implement DistanceCNN.forward according to the handout"""
        raise NotImplementedError('TODO CNN: implement DistanceCNN.forward according to the handout')
