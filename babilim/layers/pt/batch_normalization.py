import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from babilim.layers.ilayer import ILayer
from babilim.core.itensor import ITensor
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce


class BatchNormalization(ILayer):
    def __init__(self):
        super().__init__(layer_type="BatchNormalization")

    @RunOnlyOnce
    def build(self, features: ITensor):
        if len(features.shape) == 2 or len(features.shape) == 3:
            self.bn = BatchNorm1d(num_features=features.shape[1])
        elif len(features.shape) == 4:
            self.bn = BatchNorm2d(num_features=features.shape[1])
        elif len(features.shape) == 5:
            self.bn = BatchNorm3d(num_features=features.shape[1])
        else:
            raise RuntimeError("Batch norm not available for other input shapes than [B,L], [B,C,L], [B,C,H,W] or [B,C,D,H,W] dimensional.")
        
        if torch.cuda.is_available():
            self.bn = self.bn.to(torch.device("cuda"))

    def call(self, features):
        self.bn.training = self.training
        return Tensor(native=self.bn(features.native))
