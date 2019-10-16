from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.statefull_object import StatefullObject


layer_name_table = {}


class ILayer(StatefullObject):
    def __init__(self, name: str, layer_type: str):
        if name not in layer_name_table:
            layer_name_table[name] = 1
            self.__name = name
        else:
            numbering = layer_name_table[name]
            self.__name = "{}_{}".format(name, numbering)
            layer_name_table[name] += 1
        self.__layer_type = layer_type

    def __call__(self, *args, **kwargs) -> Any:
        # ensure that call gets called with ITensor objects but the caller can use native tensors.
        args, wrapped_args = self._wrapper.wrap(args)
        kwargs, wrapped_kwargs = self._wrapper.wrap(kwargs)
        self.build(*args, **kwargs)
        result = self.call(*args, **kwargs)
        if wrapped_args or wrapped_kwargs:
            return self._wrapper.unwrap(result)
        else:
            return result

    def build(self, *args, **kwargs) -> None:
        pass

    def call(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Every layer must implement this method.")

    @property
    def name(self):
        return self.__name
    
    @property
    def layer_type(self):
        return self.__layer_type

    @property
    def submodules(self):
        modules = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, ILayer):
                modules.append(v)
                modules.append(v.submodules)
        return modules

_LAYER_REGISTRY: Dict[str, Dict[str, ILayer]] = defaultdict(dict)

def register_layer(name: str, backend: str = None) -> Callable:
    def register_layer_decorator(layer):
        if backend is None:
            _LAYER_REGISTRY[TF_BACKEND][name] = layer
            _LAYER_REGISTRY[PYTORCH_BACKEND][name] = layer
        else:
            _LAYER_REGISTRY[backend][name] = layer
        return layer
    return register_layer_decorator


def get_layer(name: str) -> ILayer:
    if name not in _LAYER_REGISTRY[babilim.get_backend()]:
        raise RuntimeError("Layer {} was never registered. Did you forget to import the file in which it gets defined? Or annotating it with @register_layer(...)?".format(name))
    return _LAYER_REGISTRY[babilim.get_backend()][name]
