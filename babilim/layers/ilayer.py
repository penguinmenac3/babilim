from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor, TensorWrapper


layer_name_table = {}


class ILayer(object):
    def __init__(self, name: str, layer_type: str):
        if name not in layer_name_table:
            layer_name_table[name] = 1
            self.__name = name
        else:
            numbering = layer_name_table[name]
            self.__name = "{}_{}".format(name, numbering)
            layer_name_table[name] += 1
        self.__layer_type = layer_type
        self.__wrapper = TensorWrapper()

    def __call__(self, *args, **kwargs) -> Any:
        # ensure that call gets called with ITensor objects but the caller can use native tensors.
        args, wrapped_args = self.__wrapper.wrap(args)
        kwargs, wrapped_kwargs = self.__wrapper.wrap(kwargs)
        self.build(*args, **kwargs)
        result = self.call(*args, **kwargs)
        if wrapped_args or wrapped_kwargs:
            return self.__wrapper.unwrap(result)
        else:
            return result

    def build(self, *args, **kwargs) -> None:
        raise NotImplementedError("Every layer must implement this method.")

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
    
    @property
    def variables(self):
        all_vars = []
        extra_vars = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, str):
                pass
            elif isinstance(v, Dict):
                for k in v:
                    x = v[k]
                    if isinstance(x, ILayer):
                        all_vars.extend(x.variables)
                    if isinstance(x, ITensor):
                        all_vars.append(x)
                    if self.__wrapper.is_variable(x):
                        all_vars.append(self.__wrapper.wrap_variable(x, name=self.name + "/unnamed"))
            elif isinstance(v, Iterable):
                for x in v:
                    if isinstance(x, ILayer):
                        all_vars.extend(x.variables)
                    if isinstance(x, ITensor):
                        all_vars.append(x)
                    if self.__wrapper.is_variable(x):
                        all_vars.append(self.__wrapper.wrap_variable(x, name=self.name + "/unnamed"))
            elif isinstance(v, ILayer):
                all_vars.extend(v.variables)
            elif isinstance(v, ITensor):
                all_vars.append(v)
            elif self.__wrapper.is_variable(v):
                all_vars.append(self.__wrapper.wrap_variable(v, name=self.name + "/" + k))
            elif isinstance(v, object):
                for x in getattr(v, '_parameters', {}):
                    if isinstance(v._parameters[x], ILayer):
                        all_vars.extend(x.variables)
                    if isinstance(v._parameters[x], ITensor):
                        all_vars.append(x)
                    if self.__wrapper.is_variable(v._parameters[x]):
                        extra_vars.append(self.__wrapper.wrap_variable(v._parameters[x], name=self.name + "/" + x))
                for x in getattr(v, '__dict__', {}):
                    if isinstance(v.__dict__[x], ILayer):
                        all_vars.extend(x.variables)
                    if isinstance(v.__dict__[x], ITensor):
                        all_vars.append(x)
                    if self.__wrapper.is_variable(v.__dict__[x]):
                        extra_vars.append(self.__wrapper.wrap_variable(v.__dict__[x], name=self.name + "/" + x))
        if len(all_vars) == 0:
            all_vars.extend(extra_vars)
        return all_vars

    @property
    def trainable_variables(self):
        all_vars = self.variables
        train_vars = []
        for v in all_vars:
            if v.trainable:
                train_vars.append(v)
        return train_vars

    @property
    def trainable_variables_native(self):
        all_vars = self.trainable_variables
        train_vars = []
        for v in all_vars:
            train_vars.append(v.native)
        return train_vars

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
