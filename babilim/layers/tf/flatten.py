from tensorflow.keras.layers import Flatten as _Flatten

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class Flatten(ILayer):
    def __init__(self):
        super().__init__(layer_type="Flatten")
        self.flatten = _Flatten()

    @RunOnlyOnce
    def build(self, features):
        self.flatten.build(features.shape)

    def call(self, features):
        return Tensor(native=self.flatten(features.native))
