import math
from torch.nn import Conv2d as _Conv2d
from torch.nn.init import orthogonal_

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_pt import Tensor
from babilim.annotations import RunOnlyOnce


class Conv2D(ILayer):
    def __init__(self, filters, kernel_size, name, padding=None, stride=None, dilation=None, kernel_initializer=None):
        super().__init__(name=name, layer_type="Conv2D")
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

    @RunOnlyOnce
    def build(self, features):
        in_channels = features.shape[1]
        self.conv = _Conv2d(in_channels, self.filters, self.kernel_size, self.stride, self.padding, self.dilation)
        self.conv.weight.data = self.kernel_initializer(self.conv.weight.data)
        self.weight = Tensor(data=None, trainable=True, native=self.conv.weight, name=self.name + "/kernel")
        self.bias = Tensor(data=None, trainable=True, native=self.conv.bias, name=self.name + "/bias")

    def call(self, features):
        return Tensor(native=self.conv(features.native))
