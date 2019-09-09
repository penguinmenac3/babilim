import math
import tensorflow as tf
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.annotations import RunOnlyOnce


class Conv2D(ILayer):
    def __init__(self, filters, kernel_size, name, padding=None, strides=None, dilation_rate=None, kernel_initializer=None):
        super().__init__(name=name, layer_type="Conv2D")
        if kernel_initializer is None:
            kernel_initializer = Orthogonal()
        if padding is None:
            padding = "same"
        self.conv = _Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
                                  padding=padding, activation="relu", kernel_initializer=kernel_initializer)

    @RunOnlyOnce
    def build(self, features):
        self.conv.build(features.shape)
        self.weight = Tensor(data=None, trainable=True, native=self.conv.kernel, name=self.name + "/kernel")
        self.bias = Tensor(data=None, trainable=True, native=self.conv.bias, name=self.name + "/bias")

    def call(self, features):
        return Tensor(native=self.conv(features.native))
