from tensorflow.keras.layers import Dense

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.layers.tf.activation import Activation


class Linear(ILayer):
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
