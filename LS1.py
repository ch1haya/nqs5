import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
import complexnet as cn
import torch.optim as optim
import copy
import sys
import sr

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

L = 2
NP = L * L
NSAMPLE = 1024
U = 10
J = 1

eloc_divide = 1
eloc_samples = NSAMPLE // eloc_divide
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

    def forward(self, x):
        x = x.view(-1, 1, L, L)
        for n in range(Net.nlayer):
            x = F.pad(x, (0, 1, 0, 1),mode='circular')  # padding for 2x2
            x = self.crelu(self.conv_layers[n](x))
        x = x.view(-1, L*L*Net.nc[Net.nlayer-1])
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

class SampledState:
    thermalization_n = 1024

    def __init__(self):
        self.num = np.zeros((NSAMPLE, L, L), dtype=np.int32)
        for i in range(NSAMPLE):
            for j in range(NP):
                self.num[i][(j//L)%L][j%L] += 1

    def try_flip(self, net):
        num_tmp = np.copy(self.num)
        idx = np.arange(NSAMPLE)
        x0 = np.random.randint(L, size=NSAMPLE)
        y0 = np.random.randint(L, size=NSAMPLE)
        x1 = np.random.randint(L, size=NSAMPLE)
        y1 = np.random.randint(L, size=NSAMPLE)
        is_hop = (num_tmp[(idx, x0, y0)] > 0) & ((x0 != x1) | ( y0 != y1))
        num_tmp[(idx, x0, y0)] -= is_hop
        num_tmp[(idx, x1, y1)] += is_hop
        lnpsi_org = net.forward_only(self.num)
        lnpsi_tmp = net.forward_only(num_tmp)
        r = np.random.rand(NSAMPLE)
        isflip = r < np.exp(2 * (lnpsi_tmp.real - lnpsi_org.real))
        isflip = isflip.reshape(NSAMPLE, 1, 1)
        self.num = isflip * num_tmp + ~isflip * self.num

    def thermalize(self, net):
        for i in range(SampledState.thermalization_n):
            self.try_flip(net)

#-----------------------------------

eloc_mat = np.zeros((L, L, 4, L, L))
for x in range(L):
    for y in range(L):
        eloc_mat[x][y][0][x][y] = -1
        eloc_mat[x][y][0][(x+1)%L][y] = 1
        eloc_mat[x][y][1][x][y] = -1
        eloc_mat[x][y][1][(x-1+L)%L][y] = 1
        eloc_mat[x][y][2][x][y] = -1
        eloc_mat[x][y][2][x][(y+1)%L] = 1
        eloc_mat[x][y][3][x][y] = -1
        eloc_mat[x][y][3][x][(y-1+L)%L] = 1

def LocalEnergy(net, state):
    eloc = np.empty(NSAMPLE, dtype=np.complex64)
    lnpsi = net.forward_only(state.num)
    for k in range(eloc_divide):
        st = np.zeros((eloc_samples, L, L, 4, L, L))
        num_part = state.num[k*eloc_samples:(k+1)*eloc_samples,:,:]
        st += num_part.reshape(eloc_samples, 1, 1, 1, L, L)
        is_hop = (num_part > 0).reshape(eloc_samples, L, L, 1, 1, 1)
        st += is_hop * eloc_mat
        st = st.reshape(eloc_samples * L * L * 4, L, L)
        lnpsi2 = net.forward_only(st).reshape(eloc_samples, L, L, 4)
        lnpsi_part = lnpsi[k*eloc_samples:(k+1)*eloc_samples].reshape(
            eloc_samples, 1, 1)
        onsite = np.sum(num_part * (num_part - 1), axis=(1,2))
        hop = np.sum(np.sqrt(num_part * (np.roll(num_part, -1, axis=1) + 1))
                     * np.exp(lnpsi2[:,:,:,0] - lnpsi_part), axis=(1,2)) \
            + np.sum(np.sqrt(num_part * (np.roll(num_part, 1, axis=1) + 1))
                     * np.exp(lnpsi2[:,:,:,1] - lnpsi_part), axis=(1,2)) \
            + np.sum(np.sqrt(num_part * (np.roll(num_part, -1, axis=2) + 1))
                     * np.exp(lnpsi2[:,:,:,2] - lnpsi_part), axis=(1,2)) \
            + np.sum(np.sqrt(num_part * (np.roll(num_part, 1, axis=2) + 1))
                     * np.exp(lnpsi2[:,:,:,3] - lnpsi_part), axis=(1,2))
        eloc[k*eloc_samples:(k+1)*eloc_samples] = 0.5 * U * onsite - J * hop
    return eloc

pqn = 100
def DeltaN(net, state):
    ret = 0
    for n in range(pqn):
        for i in range(32):
            state.try_flip(net)
        var = np.var(state.num, axis=0)
        ret += np.mean(var)
    return ret / pqn

def Energy(net, state):
    ret = 0
    for n in range(pqn):
        for i in range(32):
            state.try_flip(net)
        ret += LocalEnergy(net, state).mean().real
    return ret / pqn

def Fidelity(net1, state1, net2, lam):
    state_tmp = copy.deepcopy(state1)
    a2_mean = 0
    f = 0
    f3 = 0
    for n in range(pqn):
        for i in range(32):
            state_tmp.try_flip(net1)
        logpsi_old = net1.forward_only(state_tmp.num)
        logpsi_new = net2.forward_only(state_tmp.num)
        a = np.exp(logpsi_new - logpsi_old)
        eloc = LocalEnergy(net1, state_tmp)
        a2_mean += (np.abs(a)**2).mean()
        f += ((1 + lam.conj() * 1j * dt * eloc.conj()) * a).mean()
        f3 += (np.abs(1 - 1j * dt * lam * eloc)**2).mean()
    a2_mean /= pqn
    f /= pqn
    f3 /= pqn
    return np.abs(f)**2 / (a2_mean * f3)

# -------------- main -----------------

net = Net()
net = net.to(device)


cn.add_hooks(net)
cn.disable_hooks()

optimizer = optim.Adam(net.parameters())

state = SampledState()
state.thermalize(net)

U = 10
gamma = 0.001

def main(i):
    for counter in range(i):
        for i in range(32):
            state.try_flip(net)
        eloc = LocalEnergy(net, state)
        ene = eloc.mean()
        eloc = torch.from_numpy(eloc).to(device)
        sr.StochasticReconfiguration(net, state.num, eloc, gamma)
        print(counter, ene, file=sys.stderr)
    print("finish")
    PATH = 'learnedst'
    torch.save(net.state_dict(),PATH)

def main2(i):
    for counter in range(i):
        for i in range(32):
            state.try_flip(net)
        eloc = LocalEnergy(net, state)
        ene = eloc.mean()
        eloc = torch.from_numpy(eloc).to(device)
        inputs = torch.from_numpy(state.num.astype(np.complex64)).to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = (torch.conj(outputs) * (eloc - ene)).mean().real
        loss.backward()
        optimizer.step()

        print(counter, ene, file=sys.stderr)
    print("finish")
    PATH = 'learnedst'
    torch.save(net.state_dict(),PATH)

