__version__ = "0.0.1"
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

__all__ = ['status', 'info', 'warn', 'error', 'set_backend', 'get_backend', 'is_backend']

import os
import time as __time
import datetime as __datetime


PHASE_TRAIN = "train"
PHASE_VALIDATION = "val"
PHASE_TRAINVAL = "trainval"
PHASE_TEST = "test"

DEBUG_VERBOSITY = False
PRINT_STATUS = True
PRINT_INFO = True
PRINT_WARN = True
PRINT_ERROR = True

PYTORCH_BACKEND = "pytorch"
TF_BACKEND = "tf2"

_backend = PYTORCH_BACKEND
_logfile = None


def get_logfile(path="", time_suffix=True):
    global _logfile
    if _logfile is None:
        _logfile = path
        if time_suffix:
            _logfile += __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d_%H.%M.%S')
        _logfile += ".txt"
    return _logfile


def status(msg: str, end: str= "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Babilim internally uses this for all its log messages.

    .. code-block:: python

        from babilim import tprint
        tprint("This is a log message.")
        # [yyyy-mm-dd HH:MM:SS] This is a log message.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_STATUS:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print("\r[{}] STAT {}".format(time_stamp, msg), end=end)


def info(msg: str, end: str= "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Babilim internally uses this for all its log messages.

    .. code-block:: python

        from babilim import tprint
        tprint("This is a log message.")
        # [yyyy-mm-dd HH:MM:SS] This is a log message.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_INFO:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] INFO {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        with open(get_logfile(), "a") as f:
            f.write(data + "\n")


def warn(msg: str, end: str = "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Babilim internally uses this for all its log messages.

    .. code-block:: python

        from babilim import tprint
        tprint("This is a log message.")
        # [yyyy-mm-dd HH:MM:SS] This is a log message.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_WARN:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] WARN {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        with open(get_logfile(), "a") as f:
            f.write(data + "\n")


def error(msg: str, end: str = "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Babilim internally uses this for all its log messages.

    .. code-block:: python

        from babilim import tprint
        tprint("This is a log message.")
        # [yyyy-mm-dd HH:MM:SS] This is a log message.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_ERROR:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] ERROR {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        with open(get_logfile(), "a") as f:
            f.write(data + "\n")


def set_backend(backend: str, gpu: int = None):
    """
    Set the backend which babilim uses.

    Should be either babilim.PYTORCH_BACKEND or babilim.TF_BACKEND.

    .. code-block:: python

        import babilim
        babilim.set_backend(babilim.PYTORCH_BACKEND)
        # or
        babilim.set_backend(babilim.TF_BACKEND)
    
    :param backend: The backend which should be used.
    :param gpu: (Optional) Set the gpu that should be used (cuda visible devices).
    :raises RuntimeError: When the backend is invalid or unknown.
    """
    global _backend
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    info("Using backend: {}-{}".format(backend, device))
    _backend = backend


def get_backend() -> str:
    """
    Get what backend is currently set.

    .. code-block:: python

        import babilim
        print(babilim.get_backend())
    
    :return: The backend string.
    :rtype: str
    """
    return _backend


def is_backend(backend: str) -> bool:
    """
    Check if the currently set backend is the one to check against.


    .. code-block:: python

        import babilim
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            # do pytorch specific stuff
            pass
    
    :param backend: The backend to check against
    :type backend: str
    :return: True if the set backend is equal to the provided backend.
    :rtype: bool
    """
    return _backend == backend
