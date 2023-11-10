import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from conductance_formula import conductance_LTP, pulse_LTP

def kaiming_uniform_init(weight, factor=2.0):
    # mode: fan_in
    mode = 'fan_in'
    
    """
    weights of linear layers: 2d
    weights of conv2d layers: 4d
    """
    if weight.dim() == 2:
        fan_in = weight.size(1)
    elif weight.dim() == 4:
        num_IFMs = weight.size(1)
        kernel_field_size = torch.prod(torch.tensor(weight[0].size()[1:4]))
        fan_in = num_IFMs * kernel_field_size
    else:
        raise NotImplementedError("Supporting only 2d, 4d weights")
    
    bound = math.sqrt(3 * factor / fan_in)
    weight.data.uniform_(-bound, bound)

def conductances_init(args, w):
    if args.fixed_normalization:
        ns = torch.tensor(1)
    else:
        max_weight = w.data.abs().max()
        ns = max_weight * args.w_dist_scale
    
    w_size = w.data.size()
    g_size = list(w_size)
    g_size.insert(0, 2)

    cdt = torch.zeros(g_size)
    d2d = torch.normal(torch.zeros(g_size), args.d2d_vari*torch.ones(g_size))
    nl_LTP = args.NonlinearityLTP * torch.ones_like(d2d) + d2d
    nl_LTD = args.NonlinearityLTD * torch.ones_like(d2d) + d2d
    if args.weight_representation == 'bi':
        cdt[0] = (args.Gmax-args.Gmin) * w.data/2 / ns + (args.Gmin+args.Gmax)/2
        cdt[1] = -(args.Gmax-args.Gmin) * w.data/2 / ns + (args.Gmin+args.Gmax)/2
        pulse = pulse_LTP(args, cdt, nl_LTP)  # 0: pusle of positive conductance, 1: pulse of negative conductance
        pulse = torch.round(pulse)
        pulse[pulse > args.MaxPulse] = args.MaxPulse
        cdt = conductance_LTP(args, pulse, nl_LTP)  # conductances updating from G-min
        weight = (cdt[0] - cdt[1]) / (args.Gmax-args.Gmin) * ns
    else:
        cdt[0] = (args.Gmax-args.Gmin)/2 * w.data / ns + (args.Gmin+args.Gmax)/2
        cdt[1] = (args.Gmin+args.Gmax)/2
        pulse = pulse_LTP(args, cdt, nl_LTP)
        pulse = torch.round(pulse)
        pulse[pulse > args.MaxPulse] = args.MaxPulse
        cdt = conductance_LTP(args, pulse, nl_LTP)  # conductances updating from G-min
        weight = (cdt[0] - cdt[1]) / ((args.Gmax-args.Gmin)/2) * ns

    return weight, cdt, nl_LTP, nl_LTD, ns


class HWConv2d(nn.Conv2d):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros', device=None, dtype=None):
        super(HWConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                       dilation, groups, bias, padding_mode, device, dtype)
        self.scale = kaiming_uniform_init(self.weight, factor=args.w_init_factor)
        self.weight.data, self.conductance, self.nonlinearity_LTP, self.nonlinearity_LTD, self.normalization_scale = conductances_init(args, self.weight)

class HWLinear(nn.Linear):
    def __init__(self, args, in_features, out_features, bias=False, device=None, dtype=None):
        super(HWLinear, self).__init__(in_features, out_features, bias, device, dtype)
        kaiming_uniform_init(self.weight, factor=args.w_init_factor)
        self.weight.data, self.conductance, self.nonlinearity_LTP, self.nonlinearity_LTD, self.normalization_scale = conductances_init(args, self.weight)