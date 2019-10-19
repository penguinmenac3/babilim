from babilim.layers.ilayer import ILayer
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce


class Flatten(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="Flatten")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=features.native.view(features.shape[0], -1))
