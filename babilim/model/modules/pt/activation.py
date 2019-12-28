from torch.nn.functional import relu

from babilim.model.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce


class Activation(Module):
    def __init__(self, activation: str):
        super().__init__(layer_type="Activation")
        if activation is None:
            self.activation = activation
        elif activation == "relu":
            self.activation = relu
        else:
            raise NotImplementedError("Activation '{}' not implemented.".format(activation))

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        if self.activation is None:
            return features
        else:
            return Tensor(native=self.activation(features.native))
