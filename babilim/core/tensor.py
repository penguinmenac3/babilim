from typing import Union, Any
import numpy as np

import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.itensor import ITensor, ITensorWrapper


def Tensor(*, data: Union[np.ndarray, Any], trainable: bool) -> ITensor:
    """
    Create a babilim tensor from a native tensor or numpy array.

    Arguments:
        data {Union[np.ndarray, Any]} -- [description]
        trainable {bool} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"unnamed"})
    
    Raises:
        RuntimeError: [description]
    
    Returns:
        ITensor -- [description]
    """
    if babilim.get_backend() == PYTORCH_BACKEND:
        from babilim.core.tensor_pt import Tensor as _Tensor
        from torch import Tensor as _PtTensor
        native = None
        if isinstance(data, _PtTensor):
            native = data
            data = None
        return _Tensor(data, trainable, native)
    elif babilim.get_backend() == TF_BACKEND:
        from babilim.core.tensor_tf import Tensor as _Tensor
        from tensorflow import Tensor as _TfTensor
        native = None
        if isinstance(data, _TfTensor):
            native = data
            data = None
        return _Tensor(data, trainable, native)
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
