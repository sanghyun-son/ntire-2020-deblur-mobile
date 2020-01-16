import os
from os import path

import torch
from torch import nn
from torch.nn import functional as F


class Baseline(nn.Module):
    '''
    Baseline residual network with global and local skip connections.
    (PyTorch version)

    Args:
        n_colors (int, default=3): The number of the color channels.
        n_feats (int, default=64): The number of the intermediate features.
    '''

    def __init__(self, n_colors=3, n_feats=64):
        super().__init__()
        self.conv_in = nn.Conv2d(n_colors, n_feats, 3, padding=1)
        self.res_1 = Residual(n_feats)
        self.res_2 = Residual(n_feats)
        self.res_3 = Residual(n_feats)
        self.res_4 = Residual(n_feats)
        self.conv_out = nn.Conv2d(n_feats, n_colors, 3, padding=1)

    def forward(self, x):
        res = self.conv_in(x)
        res = self.res_1(res)
        res = self.res_2(res)
        res = self.res_3(res)
        res = self.res_4(res)
        res = self.conv_out(res)
        return x + res


class Residual(nn.Module):
    '''
    A simple residual block without batch normalization.
    (PyTorch version)

    Args:
        n_feats (int): The number of the intermediate features
    '''

    def __init__(self, n_feats):
        super().__init__()
        args = [n_feats, n_feats, 3]
        self.conv1 = nn.Conv2d(*args, padding=1)
        self.conv2 = nn.Conv2d(*args, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    m = Baseline()
    dummy_state = m.state_dict()
    os.makedirs('models', exist_ok=True)
    torch.save(dummy_state, path.join('models', 'dummy_deblur.pth'))
