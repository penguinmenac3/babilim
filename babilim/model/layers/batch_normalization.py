# AUTOGENERATED FROM: babilim/model/layers/batch_normalization.ipynb

# Cell: 0
from babilim.core.annotations import RunOnlyOnce
from babilim.core.module_native import ModuleNative


# Cell: 1
class BatchNormalization(ModuleNative):
    def __init__(self):
        """
        A batch normalization layer.
        """
        super().__init__()
        
    @RunOnlyOnce
    def _build_pytorch(self, features):
        import torch
        from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
        if len(features.shape) == 2 or len(features.shape) == 3:
            self.bn = BatchNorm1d(num_features=features.shape[1])
        elif len(features.shape) == 4:
            self.bn = BatchNorm2d(num_features=features.shape[1])
        elif len(features.shape) == 5:
            self.bn = BatchNorm3d(num_features=features.shape[1])
        else:
            raise RuntimeError("Batch norm not available for other input shapes than [B,L], [B,C,L], [B,C,H,W] or [B,C,D,H,W] dimensional.")
        
        if torch.cuda.is_available():
            self.bn = self.bn.to(torch.device("cuda"))  # TODO move to correct device
        
    def _call_pytorch(self, features):
        return self.bn(features)
    
    @RunOnlyOnce
    def _build_tf(self, features):
        from tensorflow.keras.layers import BatchNormalization as _BN
        self.bn = _BN()
        
    def _call_tf(self, features):
        return self.bn(features)