from typing import Any

import babilim
from babilim.core.logging import info
from babilim.core import Tensor, RunOnlyOnce
from babilim.core.module import Module


def _isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


class Lambda(Module):
    def __init__(self, native_module):
        """
        Wrap a native module in a module.

        :param native_module: The native model, module or function to wrap. (Must accept *args and or **kwargs and return a single tensor, a list of tensors or a dict of tensors or a named tuple)
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
