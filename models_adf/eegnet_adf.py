##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/aliasvishnu/EEGNet
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Feature Extractor """
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from contrib import adf

def keep_variance(x, min_variance):
    return x+min_variance

class EEGNet_adf(nn.Module):

    def __init__(self, noise_variance=1e-2, min_variance=1e-3, initialize_msra=False):
        super(EEGNet_adf, self).__init__()

        self.keep_variance_fn=lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance
        self.ReLU = adf.ReLU(keep_variance_fn=self.keep_variance_fn)
        #self.permute=adf.permute((0,3,1,2))
        self.conv1 = adf.Conv2d(1, 16, (1, 22), padding = 0, keep_variance_fn=self.keep_variance_fn)
        self.batchnorm1 = adf.BatchNorm2d(16, False, keep_variance_fn=self.keep_variance_fn)
        # Layer 2
        self.padding1 = adf.ZeroPad2d((16, 17, 0, 1), keep_variance_fn=self.keep_variance_fn)
        self.conv2 = adf.Conv2d(1, 4, (2, 32), keep_variance_fn=self.keep_variance_fn)
        self.batchnorm2 = adf.BatchNorm2d(4, False, keep_variance_fn=self.keep_variance_fn)
        self.pooling2 = adf.MaxPool2d()
        
        # Layer 3
        self.padding2 = adf.ZeroPad2d((2, 1, 4, 3), keep_variance_fn=self.keep_variance_fn)
        self.conv3 = adf.Conv2d(4, 4, (8, 4), keep_variance_fn=self.keep_variance_fn)
        self.batchnorm3 = adf.BatchNorm2d(4, False, keep_variance_fn=self.keep_variance_fn)
        self.pooling3 = adf.MaxPool2d_2()
        self.linear = adf.Linear(4*2*25, 4, keep_variance_fn=self.keep_variance_fn)

    def forward(self, x):

        inputs_mean=x
        inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        x = inputs_mean, inputs_variance

        x = self.ReLU(*self.conv1(*x))
        x = self.batchnorm1(*x)
        #x = self.permute(*x)
        x_mean = x[0].permute(0, 3, 1, 2)
        x_var = x[1].permute(0, 3, 1, 2)
        x=x_mean, x_var
        # Layer 2
        x = self.padding1(*x)
        x = self.ReLU(*self.conv2(*x))
        x = self.batchnorm2(*x)

        x = self.pooling2(*x)
        x = self.pooling2(*x)

        # Layer 3
        x = self.padding2(*x)
        x = self.ReLU(*self.conv3(*x))
        x = self.batchnorm3(*x)
        x = self.pooling3(*x)
        
        x_mean = x[0].view(-1, 4*2*25)
        x_var=x[1].view(-1, 4*2*25)
        x = x_mean, x_var
        x = self.linear(*x)

        return x

        
