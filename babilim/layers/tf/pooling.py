import math
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D as _MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D as _MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D as _GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling1D as _GlobalAveragePooling1D

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.annotations import RunOnlyOnce


class MaxPooling2D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="MaxPooling2D")
        self.pool = _MaxPooling2D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))

class MaxPooling1D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="MaxPooling1D")
        self.pool = _MaxPooling1D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))

class GlobalAveragePooling2D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="GlobalAveragePooling2D")
        self.pool = _GlobalAveragePooling2D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))

class GlobalAveragePooling1D(ILayer):
    def __init__(self, name):
        super().__init__(name=name, layer_type="GlobalAveragePooling1D")
        self.pool = _GlobalAveragePooling1D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))
