from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import inspect

import numpy as np

import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND, warn
from babilim.core import StatefullObject, RunOnlyOnce, Tensor
from babilim.data import Dataset


class Module(StatefullObject):
    def __init__(self, layer_type: str):
        super().__init__()
        self.initialized_module = False
        self.__layer_type = layer_type

    def __call__(self, *args, **kwargs) -> Any:
        # ensure that call gets called with ITensor objects but the caller can use native tensors.
        args, wrapped_args = self._wrapper.wrap(args)
        kwargs, wrapped_kwargs = self._wrapper.wrap(kwargs)
        self.build(*args, **kwargs)
        result = self.call(*args, **kwargs)
        parent_dict = inspect.stack()[1][0].f_locals
        if "self" in parent_dict:
            parent = parent_dict["self"]
            self._register_params(parent)
        if wrapped_args or wrapped_kwargs:
            return self._wrapper.unwrap(result)
        else:
            return result

    def initialize(self, dataset: Dataset):
        if not self.initialized_module:
            if babilim.DEBUG_VERBOSITY:
                babilim.info("Build Model")
            self.initialized_module = True
            dataloader = dataset.to_dataloader()
            features, _ = next(iter(dataloader))
            self(**features._asdict())

    def predict(self, **kwargs):
        """
        Pass in single training examples as numpy arrays.
        And predict the value without gradients.
        Should be used for testing and evaluation.

        If your network has eval modes you need to set them manually.

        The array must not have batch dimension.

        :param kwargs: The parameters to feed the network as a single example.
        :return: The output for a single example.
        """
        kwargs = {k: np.array([kwargs[k]]) for k in kwargs.keys() if isinstance(kwargs[k], np.ndarray)}
        kwargs = {k: Tensor(data=kwargs[k], trainable=False) for k in kwargs.keys()}

        preds = self.__call__(**kwargs)
        tmp = preds._asdict()
        tmp = {k: tmp[k].numpy()[0] for k in tmp.keys()}
        preds = type(preds)(**tmp)
        return preds

    def build(self, *args, **kwargs) -> None:
        pass

    def call(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Every modules must implement this method.")

    @property
    def layer_type(self):
        return self.__layer_type

    @property
    def submodules(self):
        modules = []
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, Module):
                modules.append(v)
                modules.append(v.submodules)
        return modules

    def modules(self):
        """
        Returns an iterator over all modules in the network.
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """
        A named list of all modules.
        """
        modules = {}
        for k in self.__dict__:
            v = self.__dict__[k]
            if isinstance(v, Module):
                modules[prefix + "/" + k] = v
                modules.update(**v.named_modules(memo, prefix))
        return modules

    @RunOnlyOnce
    def _register_params(self, module):
        """
        Allows registration of the parameters with a native module.

        This makes the parameters of a babilim modules available to the native modules.
        When using a babilim modules in a native modules, use this function and pass the native module as a parameter.

        This function works by adding all trainable_variables to the module you pass.
        Warning: You need to build the babilim modules before calling this function. Building can be done by calling for example.

        Here is a pytorch example:

        .. code-block:: python

            import torch
            from torch.nn import Module
            from babilim.modules import Linear


            class MyModule(Module):
                def __init__(self):
                    super().__init__()
                    self.linear = Linear(10)

                def forward(self, features):
                    result = self.linear(features)
                    self.linear.register_params(self)
                    return result

        :param module: The native module on which parameters of this modules should be registered.
        """
        if babilim.is_backend(PYTORCH_BACKEND):
            from torch.nn import Module
            if isinstance(module, Module):
                myname = "_error_"
                for var in module.__dict__:
                    if module.__dict__[var] == self:
                        myname = var
                    if isinstance(module.__dict__[var], list) and self in module.__dict__[var]:
                        myname = "{}/{}".format(var, module.__dict__[var].index(self))

                # Register self as pytorch module.
                module._modules[myname] = self

                for name, param in self.named_variables.items():
                    if param.trainable:
                        module.register_parameter(myname + name, param.native)
                    else:
                        module.register_buffer(myname + name, param.native)
        else:
            warn("Not implemented for tf2 but I think it is not required.")
