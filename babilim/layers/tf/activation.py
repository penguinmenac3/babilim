import math
import tensorflow as tf
from tensorflow.keras.layers import Activation as _Activation

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.annotations import RunOnlyOnce


class ReLU(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="ReLU")
        self.activation = _Activation("relu")

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.activation(features.native))
