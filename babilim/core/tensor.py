import numpy as np

import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.itensor import ITensor, ITensorWrapper

def Tensor(*, data: np.ndarray, trainable: bool, native=None, order_flipped: bool = False) -> ITensor:
    if babilim.get_backend() == PYTORCH_BACKEND:
        from babilim.core.tensor_pt import Tensor as _Tensor
        return _Tensor(data, trainable, native, order_flipped)
    elif babilim.get_backend() == TF_BACKEND:
        from babilim.core.tensor_tf import Tensor as _Tensor
        return _Tensor(data, trainable, native, order_flipped)
    else:
        raise RuntimeError("No variable implementation for this backend was found. (backend={})".format(babilim.get_backend()))

def TensorWrapper() -> ITensorWrapper:
    if babilim.get_backend() == PYTORCH_BACKEND:
        from babilim.core.tensor_pt import TensorWrapper as _TensorWrapper
        return _TensorWrapper()
    elif babilim.get_backend() == TF_BACKEND:
        from babilim.core.tensor_tf import TensorWrapper as _TensorWrapper
        return _TensorWrapper()
    else:
        raise RuntimeError("No variable implementation for this backend was found. (backend={})".format(babilim.get_backend()))
