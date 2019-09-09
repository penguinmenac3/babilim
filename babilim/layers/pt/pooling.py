import math
from torch.nn.functional import max_pool2d as _MaxPooling2D
from torch.nn.functional import max_pool1d as _MaxPooling1D
from torch.nn.functional import avg_pool2d as _AveragePooling2D
from torch.nn.functional import avg_pool1d as _AveragePooling1D

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_pt import Tensor
from babilim.annotations import RunOnlyOnce


class MaxPooling2D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="MaxPooling2D")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling2D(features.native, (2, 2)))

class MaxPooling1D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="MaxPooling1D")
        self.pool = _MaxPooling1D(kernel_size=2, stride=2)

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling1D(features.native, 2))

class GlobalAveragePooling2D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="GlobalAveragePooling2D")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_AveragePooling2D(features.native, features.native.size()[2:]))

class GlobalAveragePooling1D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="GlobalAveragePooling1D")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_AveragePooling1D(features.native, features.native.size()[2:]))
