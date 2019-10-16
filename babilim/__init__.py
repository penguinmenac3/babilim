# MIT License
#
# Copyright (c) 2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = ['PHASE_TRAIN', 'PHASE_VALIDATION', 'PHASE_TRAINVAL',
           'PHASE_TEST', 'PYTORCH_BACKEND', 'TF_BACKEND', 'set_backend', 'get_backend', 'is_backend', 'ITensor', 'Tensor', 'RunOnlyOnce']

PHASE_TRAIN = "train"
PHASE_VALIDATION = "val"
PHASE_TRAINVAL = "trainval"
PHASE_TEST = "test"

PYTORCH_BACKEND = "pytorch"
TF_BACKEND = "tf2"

_backend = PYTORCH_BACKEND

def set_backend(backend: str):
    """
    Set the backend which babilim uses.

    Should be either babilim.PYTORCH_BACKEND or babilim.TF_BACKEND.
    
    :param backend: The backend which should be used.
    :type backend: str
    :raises RuntimeError: When the backend is invalid or unknown.
    """
    global _backend
    if backend not in [PYTORCH_BACKEND, TF_BACKEND]:
        raise RuntimeError("Unknown backend selected: {}".format(backend))
    print("Using backend: {}".format(backend))
    _backend = backend

def get_backend() -> str:
    """foobar
    
    :return: [description]
    :rtype: str
    """
    return _backend

def is_backend(backend: str) -> bool:
    """Foo
    
    :param backend: [description]
    :type backend: str
    :return: [description]
    :rtype: bool
    """
    return _backend == backend

from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor
from babilim.annotations import RunOnlyOnce
