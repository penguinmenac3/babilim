import math
from torch.nn.functional import relu

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_pt import Tensor
from babilim.annotations import RunOnlyOnce


class ReLU(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="ReLU")
        self.activation = relu

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.activation(features.native))
