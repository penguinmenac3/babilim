import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init

import torch
import torch.nn as nn

from babilim.core.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.pt.activation import Activation


class Linear(Module):
    def __init__(self, out_features, activation):
        super().__init__()
        self.out_features = out_features
        self.activation = Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        in_features = features.shape[-1]
        self.linear = PtLinear(in_features, self.out_features)
        self.weight = Tensor(data=None, trainable=True, native=self.linear.weight)
        self.bias = Tensor(data=None, trainable=True, native=self.linear.bias)
        if torch.cuda.is_available():
            self.linear = self.linear.to(torch.device("cuda"))

    def call(self, features):
        return self.activation(Tensor(native=self.linear(features.native)))


class PtLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(PtLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
