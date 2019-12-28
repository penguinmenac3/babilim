from typing import Tuple, Iterable, Sequence, List, Dict, Optional, Union, Any

from babilim import PYTORCH_BACKEND, TF_BACKEND, is_backend, get_backend
from babilim.model.module import Module

from babilim.model.modules.common.sequential import Sequential
from babilim.model.modules.common.module_wrapper import Lambda


# *******************************************************
# Various (Conv, Linear, BatchNorm)
# *******************************************************


def Flatten() -> Module:
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.flatten import Flatten as _Flatten
        return _Flatten()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.flatten import Flatten as _Flatten
        return _Flatten()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def Linear(out_features: int, activation=None) -> Module:
    """A simple linear modules.

    It computes Wx+b with no activation funciton.
    
    Arguments:
        out_features {int} -- The number of output features.
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"Linear"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.linear import Linear as _Linear
        return _Linear(out_features, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.linear import Linear as _Linear
        return _Linear(out_features, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def Conv1D(filters: int, kernel_size: int, padding: Optional[str] = None, strides: int = 1,
           dilation_rate: int = 1, kernel_initializer: Optional[Any] = None, activation=None) -> Module:
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.conv import Conv1D as _Conv1D
        return _Conv1D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.conv import Conv1D as _Conv1D
        return _Conv1D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))



def Conv2D(filters: int, kernel_size: Tuple[int, int], padding: Optional[str] = None, strides: Tuple[int, int] = (1, 1),
           dilation_rate: Tuple[int, int] = (1, 1), kernel_initializer: Optional[Any] = None, activation=None) -> Module:
    """A simple 2d convolution modules.

    TODO stride, dilation, etc.
    
    Arguments:
        filters {int} -- The number of filters (output channels).
        kernel_size {Tuple[int, int]} -- The kernel size of the filters (e.g. (3, 3)).
        kernel_l2_weight {float} -- The weight used for l2 kernel regularization. This adds a l2 loss term for your kernel weights.
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"Conv2D"})
        padding {Optional[str]} -- The padding style, if none is provided padding is done, so that the output shape is the same as the input shape. (default: {None})
        kernel_initializer {Optional[Any]} -- An initializer for the kernel. If none is provided they will be initialized orthogonally. (default: {None})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def BatchNormalization() -> Module:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"BatchNormalization"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))

# *******************************************************
# Pooling
# *******************************************************


def GlobalMaxPooling2D() -> Module:
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalMaxPooling2D as _GlobalMaxPooling2D
        return _GlobalMaxPooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalMaxPooling2D as _GlobalMaxPooling2D
        return _GlobalMaxPooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def GlobalMaxPooling1D() -> Module:
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalMaxPooling1D as _GlobalMaxPooling1D
        return _GlobalMaxPooling1D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalMaxPooling1D as _GlobalMaxPooling1D
        return _GlobalMaxPooling1D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def MaxPooling2D() -> Module:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"MaxPooling2D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def MaxPooling1D() -> Module:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"MaxPooling1D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


def GlobalAveragePooling2D() -> Module:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"GlobalPooling2D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))

# *******************************************************
# Activation Functions
# *******************************************************


def Activation(activation: str) -> Module:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your modules. (default: {"ReLU"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A modules object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.activation import Activation as _Activation
        return _Activation(activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.activation import Activation as _Activation
        return _Activation(activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))
