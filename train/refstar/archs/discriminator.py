from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import torch.nn.utils.spectral_norm as SpectralNorm
import math

@ARCH_REGISTRY.register()
class NScaleSNDiscriminator(nn.Module):
    def __init__(self, dim = 32, Scale = 512, **kwargs):
        super(NScaleSNDiscriminator, self).__init__()
        self.model = []
        self.model.append(SNFirstResBlockDiscriminator(3, dim * 1))
        cur_dim = dim
        if Scale < 1024:
            BlockNum = 5 - int(math.log(512//Scale,2))
        else:
            BlockNum = 5 - int(math.log(512/Scale,2))
        for i in range(BlockNum):
            self.model.append(SNResBlockDiscriminator(cur_dim, min(cur_dim*2,512), stride=2))
            cur_dim = cur_dim * 2
            cur_dim = min(cur_dim,512)

        if Scale < 1024:
            self.model.append(nn.AvgPool2d(8 * Scale // 512))
        else:
            self.model.append(nn.AvgPool2d(8))

        self.models = nn.Sequential(*self.model)
        self.fc = nn.Linear(cur_dim, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x, flip=0):
        if flip:
            x = torch.flip(x, (3,))
        return self.fc(self.models(x).view(-1,512))

class SNFirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SNFirstResBlockDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.LeakyReLU(0.2),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )
    def forward(self, x):
        return self.model(x) + self.bypass(x)

class SNResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SNResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
    def forward(self, x):
        return self.model(x) + self.bypass(x)