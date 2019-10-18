import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND


def GradientTape(variables):
    """
    """
    if babilim.get_backend() == PYTORCH_BACKEND:
        from babilim.core.gradient_tape_pt import GradientTapePT
        return GradientTapePT(variables)
    elif babilim.get_backend() == TF_BACKEND:
        from babilim.core.gradient_tape_tf import GradientTapeTF
        return GradientTapeTF(variables)
    else:
        raise RuntimeError("No variable implementation for this backend was found. (backend={})".format(babilim.get_backend()))
