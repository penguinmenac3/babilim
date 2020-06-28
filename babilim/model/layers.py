# AUTOGENERATED FROM: babilim/model/layers.ipynb

# Cell: 0
from typing import Tuple, Iterable, Sequence, List, Dict, Optional, Union, Any
from babilim import PYTORCH_BACKEND, TF_BACKEND, is_backend, get_backend
from babilim.model.module import Module


# Cell: 1
def Flatten() -> Module:
    """
    Flatten a feature map into a linearized tensor.
    
    This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.
    
    :return: A module implementing the flatten layer.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.flatten import Flatten as _Flatten
        return _Flatten()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.flatten import Flatten as _Flatten
        return _Flatten()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 2
def Linear(out_features: int, activation=None) -> Module:
    """
    A simple linear layer.

    It computes Wx+b with no activation funciton.
    
    :param out_features: The number of output features.
    :param activation: The activation function that should be added after the linear layer.
    :return: A module implementing the linear layer.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.linear import Linear as _Linear
        return _Linear(out_features, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.linear import Linear as _Linear
        return _Linear(out_features, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 3
def Dense(out_features: int, activation=None) -> Module:
    """
    A simple dense layer (alias for Linear Layer).

    It computes Wx+b with no activation funciton.
    
    :param out_features: The number of output features.
    :param activation: The activation function that should be added after the dense layer.
    :return: A module implementing the dense layer.
    """
    return Linear(out_features, activation)


# Cell: 4
def Conv1D(filters: int, kernel_size: int, padding: Optional[str] = None, strides: int = 1, dilation_rate: int = 1, kernel_initializer: Optional[Any] = None, activation=None) -> Module:
    """
    A 1d convolution layer.
    
    :param filters: The number of filters in the convolution. Defines the number of output channels.
    :param kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically 1, 3 or 5 are recommended.
    :param padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
    :param stride: The offset between two convolutions that are applied. Typically 1. Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
    :param dilation_rate: The dilation rate for a convolution.
    :param kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
    :param activation: The activation function that should be added after the dense layer.
    :return: A module implementing the convolution layer.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.conv import Conv1D as _Conv1D
        return _Conv1D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.conv import Conv1D as _Conv1D
        return _Conv1D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 5
def Conv2D(filters: int, kernel_size: Tuple[int, int], padding: Optional[str] = None, strides: Tuple[int, int] = (1, 1), dilation_rate: Tuple[int, int] = (1, 1), kernel_initializer: Optional[Any] = None, activation=None) -> Module:
    """
    A 2d convolution layer.
    
    :param filters: The number of filters in the convolution. Defines the number of output channels.
    :param kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically (1,1) (3,3) or (5,5) are recommended.
    :param padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
    :param stride: The offset between two convolutions that are applied. Typically (1, 1). Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
    :param dilation_rate: The dilation rate for a convolution.
    :param kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
    :param activation: The activation function that should be added after the dense layer.
    :return: A module implementing the convolution layer.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.conv import Conv2D as _Conv2D
        return _Conv2D(filters, kernel_size, padding, strides, dilation_rate, kernel_initializer, activation=activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 6
def BatchNormalization() -> Module:
    """
    A batch normalization layer.
    
    :return: A module implementing the batch normalization layer.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.batch_normalization import BatchNormalization as _BatchNormalization
        return _BatchNormalization()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 7
def GlobalMaxPooling2D() -> Module:
    """
    A global max pooling layer.
    
    This computes the global max in W, H dimension, so that the result is of shape (B, C).
    
    :return: A module implementing the global max pooling 2d.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalMaxPooling2D as _GlobalMaxPooling2D
        return _GlobalMaxPooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalMaxPooling2D as _GlobalMaxPooling2D
        return _GlobalMaxPooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 8
def GlobalMaxPooling1D() -> Module:
    """
    A global max pooling layer.
    
    This computes the global max in N dimension (B, N, C), so that the result is of shape (B, C).
    
    :return: A module implementing the global max pooling 1d.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalMaxPooling1D as _GlobalMaxPooling1D
        return _GlobalMaxPooling1D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalMaxPooling1D as _GlobalMaxPooling1D
        return _GlobalMaxPooling1D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 9
def MaxPooling2D() -> Module:
    """
    A 2x2 max pooling layer.
    
    Computes the max of a 2x2 region with stride 2.
    This halves the feature map size.
    
    :return A module implementing the 2x2 max pooling.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import MaxPooling2D as _MaxPooling2D
        return _MaxPooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 10
def MaxPooling1D() -> Module:
    """
    A max pooling layer.
    
    Computes the max of a 2 region with stride 2.
    This halves the feature map size.
    
    :return A module implementing the 2 max pooling.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import MaxPooling1D as _MaxPooling1D
        return _MaxPooling1D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 11
def GlobalAveragePooling2D() -> Module:
    """
    A global average pooling layer.
    
    This computes the global average in W, H dimension, so that the result is of shape (B, C).
    
    :return: A module implementing the global average pooling 2d.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalAveragePooling2D as _GlobalAveragePooling2D
        return _GlobalAveragePooling2D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 12
def GlobalAveragePooling1D() -> Module:
    """
    A global average pooling layer.
    
    This computes the global average in N dimension (B, N, C), so that the result is of shape (B, C).
    
    :return: A module implementing the global average pooling 1d.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.pooling import GlobalAveragePooling1D as _GlobalAveragePooling1D
        return _GlobalAveragePooling1D()
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.pooling import GlobalAveragePooling1D as _GlobalAveragePooling1D
        return _GlobalAveragePooling1D()
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))


# Cell: 13
def Activation(activation: str) -> Module:
    """
    Supports the activation functions.
    
    :param activation: A string specifying the activation function to use. (Only "relu" and None supported yet.)
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.model.modules.pt.activation import Activation as _Activation
        return _Activation(activation)
    elif is_backend(TF_BACKEND):
        from babilim.model.modules.tf.activation import Activation as _Activation
        return _Activation(activation)
    else:
        raise NotImplementedError("The backend {} is not implemented by this modules.".format(get_backend()))

# Cell: 14
def Lambda(native_module, to_gpu=True) -> Module:
    """
    Wrap a natively implemented layer into a babilim layer.
    
    This can be used to implement layers that are missing in babilim in an easy way.
    
    ```
    my_max = Lambda(tf.max)
    ```
    
    :param native_module: The native pytorch/tensorflow module that should be wrapped.
    :param to_gpu: (Optional) True if the module should be automatically be moved to the gpu. (default: True)
    """
    from babilim.model.modules.common.module_wrapper import Lambda as _Lambda
    return _Lambda(native_module, to_gpu=True)


# Cell: 15
def Sequential(*layers) -> Module:
    """
    Create a module which is a sequential order of other layers.
    
    Runs the layers in order.
    
    ```python
    my_seq = Sequential(layer1, layer2, layer3)
    ```
    
    :param layers: All ordered parameters are used as layers.
    """
    from babilim.model.modules.common.sequential import Sequential as _Sequential
    return _Sequential(*layers)
