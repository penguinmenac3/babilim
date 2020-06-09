import os
import numpy as np

from babilim import is_backend, TF_BACKEND, PYTORCH_BACKEND
from babilim.core.logging import info, warn


def load_state(checkpoint_path, native_format=False):
    if native_format:
        if is_backend(PYTORCH_BACKEND):
            import torch
            return torch.load(checkpoint_path, map_location='cpu')
        else:
            raise NotImplementedError()
    else:
        data = np.load(checkpoint_path, allow_pickle=False)
        out = {}
        prefixes = list(set([key.split("/")[0] for key in list(data.keys())]))
        for prefix in prefixes:
            if prefix in data:  # primitive types
                out[prefix] = data[prefix]
            else:  # dict types
                tmp = {"{}".format("/".join(k.split("/")[1:])): data[k] for k in data if k.startswith(prefix)}
                out[prefix] = tmp
        return out


def save_state(data, checkpoint_path, native_format=False):
    if native_format:
        if is_backend(PYTORCH_BACKEND):
            import torch
            return torch.save(data, checkpoint_path)
        else:
            raise NotImplementedError()
    else:
        out = {}
        for key, value in data.items():
            if isinstance(value, dict):
                tmp = {"{}/{}".format(key, k): value[k] for k in value}
                out.update(tmp)
            elif any(isinstance(value, t) for t in [int, str, float, list]):
                out[key] = value
            else:
                raise RuntimeError("The type ({}) of {} is not allowed!".format(type(value), key))
        np.savez_compressed(checkpoint_path, **out)
