import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
import complexnet as cn
import torch.optim as optim
import copy
import sys
import sr
import LS1 as ls

L = 2

device = torch.device("cuda:0")

class Net(nn.Module):
    nlayer = 2
    nc = (8, 8)
    fs = (2, 2)

    def __init__(self):
        super(Net, self).__init__()
        conv_layers = []
        for n in range(Net.nlayer):
            in_chan = 1 if n == 0 else Net.nc[n-1]
            conv_layers.append(cn.ComplexConv2d(in_chan, Net.nc[n], Net.fs[n]))
        self.conv_layers = nn.ModuleList(conv_layers)
        #self.fc = cn.ComplexLinear(L*L*Net.nc[Net.nlayer-1], 2, bias=False)
        self.fc = cn.ComplexLinear(L*L*Net.nc[Net.nlayer-1], 1, bias=False)
        self.crelu = cn.ComplexReLU()
        self.iniact = cn.initialactivation()
        self.subact = cn.Subseqactivation()

    def forward(self, x):
        x = x.view(-1, 1, L, L)
        #x = F.pad(x, (0, 1, 0, 1),mode='circular')  # padding for 2x2
        #x = self.iniact(self.conv_layers[0](x*1e-1))
        for n in range(Net.nlayer):
            x = F.pad(x, (0, 1, 0, 1),mode='circular')  # padding for 2x2
            #x = self.subact(self.conv_layers[n](x))
            x = self.crelu(self.conv_layers[n](x))
        x = x.view(-1, L*L*Net.nc[Net.nlayer-1])
        #x = torch.cosh(x)
        x = self.fc(x)
        x = x.view(-1)
        #x = x.view(-1, 2)
        #x = x[:,0] + torch.log(x[:,1])
        return x

    def forward_only(self, x):
        x = torch.from_numpy(x.astype(np.complex64)).to(device)
        with torch.no_grad():
            output = self.forward(x)
        return output.to("cpu").detach().numpy()
