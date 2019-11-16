from tensorflow.keras.layers import BatchNormalization as _BN

from babilim.layers.ilayer import ILayer
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class BatchNormalization(ILayer):
    def __init__(self):
        super().__init__(layer_type="BatchNormalization")
        self.bn = _BN()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.bn(features.native))
