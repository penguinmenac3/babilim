from babilim.model.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce


class Flatten(Module):
    def __init__(self):
        super().__init__(layer_type="Flatten")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=features.native.view(features.shape[0], -1))
