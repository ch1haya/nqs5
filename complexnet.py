import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ComplexConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='circular'):
        super().__init__()
        self.conv_real = nn.Conv2d(n_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias,
                                   padding_mode)
        self.conv_imag = nn.Conv2d(n_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias,
                                   padding_mode)

    def forward(self, x):
        x_real_out = self.conv_real(x.real) - self.conv_imag(x.imag)
        x_imag_out = self.conv_real(x.imag) + self.conv_imag(x.real)
        return torch.complex(x_real_out, x_imag_out)

class ComplexLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_size, out_size, bias)
        self.linear_imag = nn.Linear(in_size, out_size, bias)

    def forward(self, x):
        x_real_out = self.linear_real(x.real) - self.linear_imag(x.imag)
        x_imag_out = self.linear_real(x.imag) + self.linear_imag(x.real)
        return torch.complex(x_real_out, x_imag_out)

class ComplexReLU(nn.Module):
    def forward(self, x):
        x_real_out = F.relu(x.real)
        x_imag_out = F.relu(x.imag)
        return torch.complex(x_real_out, x_imag_out)

class initialactivation(nn.Module):
    def forward(self,i):
        x = i.real
        y = i.imag
        result_re =  1/45*(x**6 -15 * x**4 * y**2 - y**6)-1/12*(x**4 - 6*x**2*y**2 + y**4)+1/2*(x**2 - y**2)
        result_im =  1/45*(6*x**5*y -20 * x**3*y**3 + 6*x*y**5)-1/12*(4 * x**3 * y - 4* x * y**3)+1/2*(2*x*y)
        return torch.complex(result_re,result_im)

class Subseqactivation(nn.Module):
    def forward(self,i):
        x = i.real
        y = i.imag
        result_re =  2/15*(x**5 - 10*x**3*y**2 + 5 * x * y **4)-1/3*(x**3 - 3 * x * y**2)+ x
        result_im =  2/15*(5 * x**4 *y + y**5) - 1/3*y*(3* x**2 * y - y**3) + y
        return torch.complex(result_re,result_im)

#--------------------------------------------------------------------

_supported_layers = ['Linear', 'Conv2d']
_hooks_disabled: bool = False

def add_hooks(model: nn.Module) -> None:
    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)

def disable_hooks() -> None:
    global _hooks_disabled
    _hooks_disabled = True

def enable_hooks() -> None:
    global _hooks_disabled
    _hooks_disabled = False

def is_supported(layer: nn.Module) -> bool:
    return _layer_type(layer) in _supported_layers

def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

def _capture_activations(layer: nn.Module, input: List[torch.Tensor],
                         output: torch.Tensor):
    if _hooks_disabled:
        return
    if not hasattr(layer, 'activations'):
        setattr(layer, 'activations', [])
    if _layer_type(layer) == 'Conv2d':
        p = layer.padding[0]
        x = F.pad(input[0].detach(), (p, p, p, p), layer.padding_mode)
    else: # 'Linear'
        x = input[0].detach()
    layer.activations.append(x)

def _capture_backprops(layer: nn.Module, _input, output):
    if _hooks_disabled:
        return
    if not hasattr(layer, 'backprops'):
        setattr(layer, 'backprops', [])
    layer.backprops.append(output[0].detach())

def clear_captures(model: nn.Module) -> None:
    for layer in model.modules():
        if hasattr(layer, 'backprops'):
            del layer.backprops
        if hasattr(layer, 'activations'):
            del layer.activations

def compute_grad1(model: nn.Module) -> None:
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops) == 2, "Multiple backprops detected, make sure to call clear_captures(model)"

        A0 = layer.activations[0]
        A1 = layer.activations[1]
        B0 = layer.backprops[0]
        B1 = layer.backprops[1]
        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B0, A1)
                    + torch.einsum('ni,nj->nij', B1, A0))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B0 + B1)

        elif layer_type == 'Conv2d':
            n = A0.shape[0]
            A0 = torch.nn.functional.unfold(A0, layer.kernel_size)
            A1 = torch.nn.functional.unfold(A1, layer.kernel_size)
            B0 = B0.reshape(n, -1, A0.shape[-1])
            B1 = B1.reshape(n, -1, A1.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', B0, A1) \
                    + torch.einsum('ijk,ilk->ijl', B1, A0)
            shape = [n] + list(layer.weight.shape)
            setattr(layer.weight, 'grad1', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B0 + B1, dim=2))
    
