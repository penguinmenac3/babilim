from tensorflow.keras.layers import Dense

from babilim.model.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.tf import Activation


class Linear(Module):
    def __init__(self, out_features, activation):
        super().__init__(layer_type="Linear")
        self.linear = Dense(out_features)
        self.activation = Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        self.linear.build(features.shape)
        self.weight = Tensor(data=None, trainable=True, native=self.linear.kernel)
        self.bias = Tensor(data=None, trainable=True, native=self.linear.bias)

    def call(self, features):
        return self.activation(Tensor(native=self.linear(features.native)))
