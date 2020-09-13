# AUTOGENERATED FROM: babilim/model/layers/fully_connected.ipynb

# Cell: 0
"""
# babilim.model.layers.fully_connected

> A simple fully connected layer (aka Linear Layer or Dense).
"""

# Cell: 1
from babilim.core.annotations import RunOnlyOnce
from babilim.core.module_native import ModuleNative
from babilim.model.layers.activation import Activation


# Cell: 2
class FullyConnected(ModuleNative):
    def __init__(self, out_features: int, activation=None):
        """
        A simple fully connected layer (aka Linear Layer or Dense).

        It computes Wx+b with optional activation funciton.

        :param out_features: The number of output features.
        :param activation: The activation function that should be added after the fc layer.
        """
        super().__init__()
        self.out_features = out_features
        self.activation = Activation(activation)
        
    @RunOnlyOnce
    def _build_pytorch(self, features):
        import torch
        from babilim.core.tensor_pt import Tensor as _Tensor
        in_features = features.shape[-1]
        self.linear = torch.nn.Linear(in_features, self.out_features)
        self.weight = _Tensor(data=None, trainable=True, native=self.linear.weight)
        self.bias = _Tensor(data=None, trainable=True, native=self.linear.bias)
        if torch.cuda.is_available():
            self.linear = self.linear.to(torch.device("cuda"))  # FIXME shouldn't this be done automatically?
        
    def _call_pytorch(self, features):
        return self.activation(self.linear(features))
    
    @RunOnlyOnce
    def _build_tf(self, features):
        from tensorflow.keras.layers import Dense
        from babilim.core.tensor_tf import Tensor as _Tensor
        self.linear = Dense(self.out_features)
        self.linear.build(features.shape)
        self.weight = _Tensor(data=None, trainable=True, native=self.linear.kernel)
        self.bias = _Tensor(data=None, trainable=True, native=self.linear.bias)

    def _call_tf(self, features):
        return self.activation(self.linear(features))
