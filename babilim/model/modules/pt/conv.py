import torch
from torch.nn import Conv2d as _Conv2d
from torch.nn import Conv1d as _Conv1d
from torch.nn.init import orthogonal_

from babilim.core.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.pt.activation import Activation


class Conv2D(Module):
    def __init__(self, filters, kernel_size, padding=None, stride=None, dilation=None, kernel_initializer=None, activation=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        if kernel_initializer is None:
            kernel_initializer = orthogonal_
        if padding == "same" or padding is None:
            px = int((kernel_size[0] - 1) / 2)
            py = int((kernel_size[1] - 1) / 2)
            padding = (px, py)
        elif padding == "none":
            padding = (0, 0)
        else:
            raise NotImplementedError("Padding {} is not implemented.".format(padding))
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.activation = Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        in_channels = features.shape[1]
        self.conv = _Conv2d(in_channels, self.filters, self.kernel_size, self.stride, self.padding, self.dilation)
        self.conv.weight.data = self.kernel_initializer(self.conv.weight.data)
        if torch.cuda.is_available():
            self.conv = self.conv.to(torch.device("cuda"))
        self.weight = Tensor(data=None, trainable=True, native=self.conv.weight)
        self.bias = Tensor(data=None, trainable=True, native=self.conv.bias)

    def call(self, features):
        return self.activation(Tensor(native=self.conv(features.native)))


class Conv1D(Module):
    def __init__(self, filters, kernel_size, padding=None, stride=None, dilation=None, kernel_initializer=None, activation=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        if kernel_initializer is None:
            kernel_initializer = orthogonal_
        if padding == "same" or padding is None:
            padding = int((kernel_size - 1) / 2)
        elif padding == "none":
            padding = 0
        else:
            raise NotImplementedError("Padding {} is not implemented.".format(padding))
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.kernel_initializer = kernel_initializer
        self.activation = Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        in_channels = features.shape[1]
        self.conv = _Conv1d(in_channels, self.filters, self.kernel_size, self.stride, self.padding, self.dilation)
        self.conv.weight.data = self.kernel_initializer(self.conv.weight.data)
        if torch.cuda.is_available():
            self.conv = self.conv.to(torch.device("cuda"))
        self.weight = Tensor(data=None, trainable=True, native=self.conv.weight)
        self.bias = Tensor(data=None, trainable=True, native=self.conv.bias)

    def call(self, features):
        return self.activation(Tensor(native=self.conv(features.native)))
