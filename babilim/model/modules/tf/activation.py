from tensorflow.keras.layers import Activation as _Activation

from babilim.model.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class Activation(Module):
    def __init__(self, activation: str):
        super().__init__(layer_type="Activation")
        if self.activation is None:
            self.activation = None
        else:
            self.activation = _Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        if self.activation is None:
            return features
        else:
            return Tensor(native=self.activation(features.native))
