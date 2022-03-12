import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_NUM_THREADS"] = "32"

import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
import complexnet as cn
import torch.optim as optim
import copy
import sys
import ls_test as ls
import timedev as timedev
import networkconfig as nc

import matplotlib
import matplotlib.pyplot as plt
import math

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

L = 2
NP = L * L
NSAMPLE = 1024
U = 10
J = 1

learn_r = 1000

eloc_divide = 1
eloc_samples = NSAMPLE // eloc_divide
device = torch.device("cuda:0")


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
        f += ((1 + lam.conj() * 1j * delta * eloc.conj()) * a).mean()
        f3 += (np.abs(1 - 1j * delta * lam * eloc)**2).mean()
    a2_mean /= pqn
    f /= pqn
    f3 /= pqn
    return np.abs(f)**2 / (a2_mean * f3)

def td_sr(time):
    counter = 0
    while True:
        for i in range(36):
            state.try_flip(net)
        eloc = LocalEnergy(net, state)
        ene = eloc.mean()
        energy.append(ene.real)
        #dw, x, sigma  = timedev.caldW(net, state.num, eloc, delta,tole)
        dw = timedev.NewtonMethod(net,state.num,eloc,delta,tole)
        #timedev.NewtonMethod(net,state.num,eloc,delta,tole)
        site = state.num
        site = site.reshape(NSAMPLE,1,L*L)
        flucs = np.var(site, axis=0)
        flucmean = np.mean(flucs)
        fluc.append(flucmean)
        print(delta*counter, ene.real, flucmean,np.mean(dw),flush=True)
        count.append(counter)
        counter += 1
        if len(count) > 4:
            if counter*delta > 2:
                print('break!')
                #plot
                fig = plt.figure()
                plt.plot(count,energy)
                plt.xlabel("t")
                plt.ylabel("<E>")
                plt.title("Energy")
                #plt.xlim(4,counter)
                plt.grid()
                plt.show()
                fig.savefig("timedeveloed_energyfixed3.png")

                fig = plt.figure()
                plt.plot(count, fluc)
                plt.xlabel("t")
                plt.ylabel("<σxy>")
                plt.title("fluctuation of Particle")
                plt.xlim(4,counter)
                plt.grid()
                plt.show()
                fig.savefig("timedeveloed_flucfixed3.png")
                break

def td_conv(time):
    counter = 0
    lam = np.array([0.04262665650265929 - 0.3946329531719595j, 0.4573733434973211 + 0.23510048799868316j, 0.04262665650265929 + 0.3946329531719595j, 0.4573733434973211 - 0.23510048799868316j])
    err = 0
    while True:
        print(counter * delta, Energy(net, state), DeltaN(net, state), err, flush=True)
        err = 0
        for b in range(4):
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            #schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.01)#usewith SGD
            schedular = optim.lr_scheduler.StepLR(optimizer,step_size = 2,gamma = 0.1) #please refine
            net_old = copy.deepcopy(net)
            for i in range(32):
                state.try_flip(net_old)
            eloc = LocalEnergy(net_old, state)
            d_eloc = torch.from_numpy(eloc).to(device)
            logpsi_old = net_old.forward_only(state.num)
            inputs = torch.from_numpy(state.num.astype(np.complex64)).to(device)

            for cnt in range(100):
            #for cnt in range(100+200+400+800):
                logpsi_new = net.forward_only(state.num)
                a = np.exp(logpsi_new - logpsi_old)
                a2_mean = (np.abs(a)**2).mean()
                f = ((1 + lam[b].conj() * 1j * delta * eloc.conj()) * a).mean()

                a = torch.from_numpy(a).to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                f2 = a * outputs * (1 + lam[b].conj() * 1j * delta * torch.conj(d_eloc))
                loss = -(f2 / f - a.abs()**2 * outputs / a2_mean).mean().real
                print(cnt, np.abs(f)**2 / (a2_mean * (np.abs(1 - 1j * delta * lam[b] * eloc)**2).mean()))
                loss.backward()
                optimizer.step()
                schedular.step()
                err += 1 - Fidelity(net_old, state, net, lam[b])
            counter += 1
            if delta*counter > time:
                break


ls.main2(learn_r)

net = nc.Net()
net.load_state_dict(torch.load('learnedst_test')) #test method

state = SampledState()

net = net.to(device)
state.thermalize(net)

cn.add_hooks(net)
cn.disable_hooks()

U = 1;
delta = 1e-5
tole = 1e-9


energy = []
kinetic = []
pot = []
count = []
energy = []
flucmean = []
fluc = []


#main

print(net)
print('original U:10')
print('to U,-t,delta,NSAMPLE',U,-J,delta,NSAMPLE)
print('site :',L)

select = 1
if select == 1:
    print('use sr')
    td_sr(2.0)
else:
    print('use conv')
    td_conv(1)


