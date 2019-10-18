from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.statefull_object import StatefullObject


class ILayer(StatefullObject):
    def __init__(self, name: str, layer_type: str):
        super().__init__(name)
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
