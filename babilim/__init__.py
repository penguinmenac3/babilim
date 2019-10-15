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

import time as __time
import datetime as __datetime


PHASE_TRAIN = "train"
PHASE_VALIDATION = "val"
PHASE_TRAINVAL = "trainval"
PHASE_TEST = "test"

PYTORCH_BACKEND = "pytorch"
TF_BACKEND = "tf2"

_backend = PYTORCH_BACKEND

def tprint(msg: str, end: str="\n"):
    time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print("\r[{}] {}".format(time_stamp, msg), end=end)

def set_backend(backend: str):
    global _backend
    if backend not in [PYTORCH_BACKEND, TF_BACKEND]:
        raise RuntimeError("Unknown backend selected: {}".format(backend))
    device = "cpu"
    if backend == PYTORCH_BACKEND:
        import torch
        if torch.cuda.is_available():
            device = "gpu"
    else:
        import tensorflow as tf
        if tf.test.is_gpu_available():
            device = "gpu"
    tprint("Using backend: {}-{}".format(backend, device))
    _backend = backend

def get_backend() -> str:
    return _backend

def is_backend(backend: str) -> bool:
    return _backend == backend

from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor
