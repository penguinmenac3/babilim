import math
import tensorflow as tf
from tensorflow.keras.layers import Dense

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.annotations import RunOnlyOnce


class Linear(ILayer):
    def __init__(self, out_features, name):
        super().__init__(name=name, layer_type="Linear")
        self.linear = Dense(out_features)

    @RunOnlyOnce
    def build(self, features):
        self.linear.build(features.shape)
        self.weight = Tensor(data=None, trainable=True, native=self.linear.kernel, order_flipped=True)
        self.bias = Tensor(data=None, trainable=True, native=self.linear.bias)

    def call(self, features):
        return Tensor(native=self.linear(features.native))
