import math
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from babilim.layers.ilayer import ILayer
from babilim.core.itensor import ITensor
from babilim.core.tensor_pt import Tensor
from babilim.annotations import RunOnlyOnce


class BatchNormalization(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="BatchNormalization")

    @RunOnlyOnce
    def build(self, features: ITensor):
        if len(features.shape) == 3:
            self.bn = BatchNorm1d(features.shape[1])
        elif len(features.shape) == 4:
            self.bn = BatchNorm2d(features.shape[1])
        elif len(features.shape) == 5:
            self.bn = BatchNorm3d(features.shape[1])
        else:
            raise RuntimeError("Batch norm not available for other input shapes than 3, 4 or 5 dimensional.")
        if self.bn.weight is not None:
            self.weight = Tensor(native=self.bn.weight)
        if self.bn.bias is not None:
            self.bias = Tensor(native=self.bn.bias)


    def call(self, features):
        return Tensor(native=self.bn(features.native))
