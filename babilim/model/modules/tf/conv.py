from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.initializers import Orthogonal

from babilim.model.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.tf.activation import Activation


class Conv2D(Module):
    def __init__(self, filters, kernel_size, padding=None, strides=None, dilation_rate=None, kernel_initializer=None, activation=None):
        super().__init__(layer_type="Conv2D")
        if kernel_initializer is None:
            kernel_initializer = Orthogonal()
        if padding is None:
            padding = "same"
        self.conv = _Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                  padding=padding, activation="relu", kernel_initializer=kernel_initializer)
        self.activation = Activation(activation)

    @RunOnlyOnce
    def build(self, features):
        self.conv.build(features.shape)
        self.weight = Tensor(data=None, trainable=True, native=self.conv.kernel)
        self.bias = Tensor(data=None, trainable=True, native=self.conv.bias)

    def call(self, features):
        return self.activation(Tensor(native=self.conv(features.native)))
