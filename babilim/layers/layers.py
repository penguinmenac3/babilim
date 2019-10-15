from typing import Tuple, Iterable, Sequence, List, Dict, Optional, Union, Any

from babilim import PYTORCH_BACKEND, TF_BACKEND, is_backend, get_backend
from babilim.layers.ilayer import ILayer, register_layer

from babilim.layers.common.sequential import Sequential
from babilim.layers.common.wrapper_layer import Lambda


# *******************************************************
# Various (Conv, Linear, BatchNorm)
# *******************************************************

@register_layer(TF_BACKEND, "Flatten")
@register_layer(PYTORCH_BACKEND, "Flatten")
def Flatten(name:str = "Flatten") -> ILayer:
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.flatten import Flatten as _Flatten
        return _Flatten(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.flatten import Flatten as _Flatten
        return _Flatten(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

@register_layer(TF_BACKEND, "Linear")
@register_layer(PYTORCH_BACKEND, "Linear")
def Linear(out_features: int, name:str ="Linear") -> ILayer:
    """A simple linear layer.

    It computes Wx+b with no activation funciton.
    
    Arguments:
        out_features {int} -- The number of output features.
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"Linear"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.linear import Linear as _Linear
        return _Linear(out_features, name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.linear import Linear as _Linear
        return _Linear(out_features, name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

@register_layer(TF_BACKEND, "Conv2D")
@register_layer(PYTORCH_BACKEND, "Conv2D")
def Conv2D(filters: int, kernel_size: Tuple[int, int], name: str = "Conv2D",
           padding: Optional[str] = None, strides: Tuple[int, int] = (1, 1),
           dilation_rate: Tuple[int, int] = (1, 1), kernel_initializer: Optional[Any] = None) -> ILayer:
    """A simple 2d convolution layer.

    TODO stride, dilation, etc.
    
    Arguments:
        filters {int} -- The number of filters (output channels).
        kernel_size {Tuple[int, int]} -- The kernel size of the filters (e.g. (3, 3)).
        kernel_l2_weight {float} -- The weight used for l2 kernel regularization. This adds a l2 loss term for your kernel weights.
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"Conv2D"})
        padding {Optional[str]} -- The padding style, if none is provided padding is done, so that the output shape is the same as the input shape. (default: {None})
        kernel_initializer {Optional[Any]} -- An initializer for the kernel. If none is provided they will be initialized orthogonally. (default: {None})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, name, padding, strides, dilation_rate, kernel_initializer)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, name, padding, strides, dilation_rate, kernel_initializer)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

@register_layer(TF_BACKEND, "BatchNormalization")
@register_layer(PYTORCH_BACKEND, "BatchNormalization")
def BatchNormalization(name:str ="BatchNormalization") -> ILayer:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"BatchNormalization"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

# *******************************************************
# Pooling
# *******************************************************

@register_layer(TF_BACKEND, "MaxPooling2D")
@register_layer(PYTORCH_BACKEND, "MaxPooling2D")
def MaxPooling2D(name:str ="MaxPooling2D") -> ILayer:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"MaxPooling2D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

@register_layer(TF_BACKEND, "MaxPooling1D")
@register_layer(PYTORCH_BACKEND, "MaxPooling1D")
def MaxPooling1D(name:str ="MaxPooling1D") -> ILayer:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"MaxPooling1D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

@register_layer(TF_BACKEND, "GlobalAveragePooling2D")
@register_layer(PYTORCH_BACKEND, "GlobalAveragePooling2D")
def GlobalAveragePooling2D(name:str ="GlobalAveragePooling2D") -> ILayer:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"GlobalPooling2D"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))

# *******************************************************
# Activation Functions
# *******************************************************

@register_layer(TF_BACKEND, "ReLU")
@register_layer(PYTORCH_BACKEND, "ReLU")
def ReLU(name:str ="ReLU") -> ILayer:
    """TODO
    
    Arguments:
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"ReLU"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.activation import ReLU as _ReLU
        return _ReLU(name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.activation import ReLU as _ReLU
        return _ReLU(name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))
