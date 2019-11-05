from babilim.core import ITensor, Tensor
import tensorflow as tf


def log(tensor: ITensor) -> ITensor:
    """
    Compute the logarithm of a tensor.

    :param tensor: The tensor of which the log should be computed.
    :return: The log tensor.
    """
    return Tensor(data=tf.log(tensor.native), trainable=True)


def clip(tensor: ITensor, min, max) -> ITensor:
    """
    Clip a tensor to a value range.

    :param tensor: The tensor which should be clipped.
    :param min: The minimum value to clip to.
    :param max: The maximum value to clip to.
    :return: The clipped tensor.
    """
    return Tensor(data=tf.clamp(tensor.native, min, max), trainable=True)  # FIXME does this work?


def pow(tensor: ITensor, exponent: ITensor) -> ITensor:
    """
    Compute the power of a tensor.

    :param tensor: The tensor of which the pow should be computed.
    :param exponent: The exponent to take the power with.
    :return: The pow tensor.
    """
    return Tensor(data=tf.pow(tensor.native, exponent.native), trainable=True)
