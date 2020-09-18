# AUTOGENERATED FROM: babilim/model/layers/native_wrapper.ipynb

# Cell: 0
"""doc
# babilim.model.layers.native_wrapper

> Wrap a layer, function or model from native into babilim.
"""

# Cell: 1
from typing import Any

import babilim
from babilim.core.logging import info
from babilim.core import Tensor, RunOnlyOnce
from babilim.core.module import Module


# Cell: 2
def _isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


# Cell: 3
class Lambda(Module):
    def __init__(self, native_module, to_gpu=True):
        """
        Wrap a natively implemented layer into a babilim layer.
    
        This can be used to implement layers that are missing in babilim in an easy way.
        
        ```
        my_lambda = Lambda(tf.max)
        ```
        
        :param native_module: The native pytorch/tensorflow module that should be wrapped.
        :param to_gpu: (Optional) True if the module should be automatically be moved to the gpu. (default: True)
        """
        super().__init__()
        self.native_module = native_module

    def _auto_device(self):
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            import torch
            self.native_module = self.native_module.to(torch.device(self.device))
            return self
    
    @RunOnlyOnce
    def build(self, *args, **kwargs) -> None:
        self._auto_device()
        build = getattr(self.native_module, "build", None)
        if callable(build):
            # Unwrap arguments
            args = [feature.native for feature in args]
            kwargs = {k: kwargs[k].native for k in kwargs}

            # Call the build
            build(*args, **kwargs)

    def call(self, *args, **kwargs) -> Any:
        """
        Do not call this directly, use `__call__`:
        ```
        my_lambda(*args, **kwargs)
        ```
        """
        # Unwrap arguments
        args = [feature.native for feature in args]
        kwargs = {k: kwargs[k].native for k in kwargs}

        # call function
        result = self.native_module(*args, **kwargs)
        # Wrap results
        if _isnamedtupleinstance(result):
            result_raw = result._asdict()
            result_raw = {k: Tensor(data=result_raw[k], trainable=True) for k in result_raw}
            return type(result)(**result_raw)
        elif isinstance(result, dict):
            result = {k: Tensor(data=result[k], trainable=True) for k in result}
        elif isinstance(result, list):
            result = [Tensor(data=res, trainable=True) for res in result]
        else:
            result = Tensor(data=result, trainable=True)
        return result

    def eval(self):
        self.train(False)

    def train(self, mode=True):
        train_fn = getattr(self.native_module, "train", None)
        if callable(train_fn):
            train_fn(mode)
