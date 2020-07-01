from tensorflow.keras.layers import Flatten as _Flatten

from babilim.core.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class Flatten(Module):
    def __init__(self):
        super().__init__()
        self.flatten = _Flatten()

    @RunOnlyOnce
    def build(self, features):
        self.flatten.build(features.shape)

    def call(self, features):
        return Tensor(native=self.flatten(features.native))
