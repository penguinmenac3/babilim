PYTORCH_BACKEND = "pytorch"
TF_BACKEND = "tf2"

_backend = PYTORCH_BACKEND

def set_backend(backend: str):
    global _backend
    if backend not in [PYTORCH_BACKEND, TF_BACKEND]:
        raise RuntimeError("Unknown backend selected.".format(backend))
    _backend = backend

def get_backend() -> str:
    return _backend

def is_backend(backend: str) -> bool:
    return _backend == backend

from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor
