from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import inspect
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND, warn
from babilim.core import StatefullObject, RunOnlyOnce


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
        self._register_params(inspect.stack()[1][0].f_locals["self"])
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

    @RunOnlyOnce
    def _register_params(self, module):
        """
        Allows registration of the parameters with a native module.

        This makes the parameters of a babilim layer available to the native layer.
        When using a babilim layer in a native layer, use this function and pass the native module as a parameter.

        This function works by adding all trainable_variables to the module you pass.
        Warning: You need to build the babilim layer before calling this function. Building can be done by calling for example.

        Here is a pytorch example:

        .. code-block:: python

            import torch
            from torch.nn import Module
            from babilim.layers import Linear


            class MyModule(Module):
                def __init__(self):
                    super().__init__()
                    self.linear = Linear(10)

                def forward(self, features):
                    result = self.linear(features)
                    self.linear.register_params(self)
                    return result

        :param module: The native module on which parameters of this layer should be registered.
        """
        if babilim.is_backend(PYTORCH_BACKEND):
            from torch.nn import Module
            if isinstance(module, Module):
                for param in self.trainable_variables:
                    module.register_parameter(param.name, param.native)
        else:
            warn("Not implemented for tf2 but I think it is not required.")
