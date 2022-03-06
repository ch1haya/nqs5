import numpy as np
import torch
import complexnet as cn
from numpy import linalg as LA
from scipy.sparse.linalg import minres
from scipy.sparse.linalg import LinearOperator


device = torch.device("cuda:0")

def get_grad1(net, sample_num):
    list = []
    for layer in net.modules():
        if not cn.is_supported(layer):
            continue
        for param in layer.parameters():
            list.append(param.grad1.reshape(sample_num, -1))
    return torch.cat(list, dim=1)

def getOk(net, inputs):
    sample_num = inputs.shape[0]
    cn.enable_hooks()

    output = net(inputs)
    output.real.sum().backward(retain_graph=True)
    cn.compute_grad1(net)
    Ok_real = get_grad1(net, sample_num)
    cn.clear_captures(net)

    output = net(inputs)
    output.imag.sum().backward(retain_graph=True)
    cn.compute_grad1(net)
    Ok_imag = get_grad1(net, sample_num)
    cn.clear_captures(net)

    cn.disable_hooks()
    return torch.complex(Ok_real, Ok_imag)

def UpdateNetworkParameters(net, delta_w):
    ptr = 0
    for layer in net.modules():
        if not cn.is_supported(layer):
            continue
        for param in layer.parameters():
            size = param.nelement()
            dw = delta_w[ptr:ptr+size].reshape(param.shape)
            ptr += size
            param.data += dw

def StochasticReconfiguration(net, inputs, eloc, gamma):
    sample_num = inputs.shape[0]
    inputs = torch.from_numpy(inputs.astype(np.complex64)).to(device)
    
    eloc = eloc.to("cpu").detach().numpy()

    Ok = getOk(net, inputs)
    Ok = Ok.to("cpu").detach().numpy()   # complex not implemented on gpu
    Ok_avg = np.mean(Ok, axis=0)
    nparam = Ok_avg.shape[0]
    
    def MultiplySkk(x):
        return np.matmul(Ok.conj().T, np.matmul(Ok, x)).real / sample_num - (Ok_avg.conj() * Ok_avg.dot(x)).real
    Fk = np.matmul(eloc, Ok.conj()) / sample_num - eloc.mean() * Ok_avg.conj()
    
    op = LinearOperator((nparam, nparam), matvec=MultiplySkk)
    x, exitCode = minres(op, Fk.real)
    dw = -gamma * x 
    dw = torch.from_numpy(dw.astype(np.float32)).to(device)
    UpdateNetworkParameters(net, dw)
