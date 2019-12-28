from babilim.core import ITensor
from babilim import is_backend, TF_BACKEND, PYTORCH_BACKEND


def log(tensor: ITensor) -> ITensor:
    """
    Compute the logarithm of a tensor.

    :param tensor: The tensor of which the log should be computed.
    :return: The log tensor.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.core.tmath.pytorch import log as _log
        return _log(tensor)
    elif is_backend(TF_BACKEND):
        from babilim.core.tmath.tf import log as _log
        return _log(tensor)


def clip(tensor: ITensor, min, max) -> ITensor:
    """
    Clip a tensor to a value range.

    :param tensor: The tensor which should be clipped.
    :param min: The minimum value to clip to.
    :param max: The maximum value to clip to.
    :return: The clipped tensor.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.core.tmath.pytorch import clip as _clip
        return _clip(tensor, min, max)
    elif is_backend(TF_BACKEND):
        from babilim.core.tmath.tf import clip as _clip
        return _clip(tensor, min, max)


def pow(tensor: ITensor, exponent: ITensor) -> ITensor:
    """
    Compute the power of a tensor.

    :param tensor: The tensor of which the pow should be computed.
    :param exponent: The exponent to take the power with.
    :return: The pow tensor.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.core.tmath.pytorch import pow as _pow
        return _pow(tensor, exponent)
    elif is_backend(TF_BACKEND):
        from babilim.core.tmath.tf import pow as _pow
        return _pow(tensor, exponent)
