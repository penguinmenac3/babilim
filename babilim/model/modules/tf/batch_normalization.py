from tensorflow.keras.layers import BatchNormalization as _BN

from babilim.core.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class BatchNormalization(Module):
    def __init__(self):
        super().__init__()
        self.bn = _BN()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.bn(features.native))
