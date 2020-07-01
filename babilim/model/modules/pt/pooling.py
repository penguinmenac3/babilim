from torch.nn.functional import max_pool2d as _MaxPooling2D
from torch.nn.functional import max_pool1d as _MaxPooling1D
from torch.nn.functional import avg_pool2d as _AveragePooling2D
from torch.nn.functional import avg_pool1d as _AveragePooling1D

from babilim.core.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.pt.flatten import Flatten


class MaxPooling2D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling2D(features.native, (2, 2)))


class MaxPooling1D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling1D(features.native, 2))

class GlobalMaxPooling2D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features: Tensor):
        shape = features.shape[1:2]
        print(shape)
        return Tensor(native=_MaxPooling2D(features.native, shape))

class GlobalMaxPooling1D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features: Tensor):
        shape = features.shape[1]
        return Tensor(native=_MaxPooling1D(features.native, shape))

class GlobalAveragePooling2D(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return self.flatten(Tensor(native=_AveragePooling2D(features.native, features.native.size()[2:])))


class GlobalAveragePooling1D(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return self.flatten(Tensor(native=_AveragePooling1D(features.native, features.native.size()[2:])))
