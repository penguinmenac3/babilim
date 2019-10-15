from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor, TensorWrapper

class StatefullObject(object):
    _wrapper = TensorWrapper()
    name = "unnamed"

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
                    if isinstance(x, StatefullObject):
                        all_vars.extend(x.variables)
                    if isinstance(x, ITensor):
                        all_vars.append(x)
                    if self._wrapper.is_variable(x):
                        all_vars.append(self._wrapper.wrap_variable(x, name=self.name + "/unnamed"))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, self.name, k))
            elif isinstance(v, Iterable):
                for x in v:
                    if isinstance(x, StatefullObject):
                        all_vars.extend(x.variables)
                    if isinstance(x, ITensor):
                        all_vars.append(x)
                    if self._wrapper.is_variable(x):
                        all_vars.append(self._wrapper.wrap_variable(x, name=self.name + "/unnamed"))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, self.name, k))
            elif isinstance(v, StatefullObject):
                all_vars.extend(v.variables)
            elif isinstance(v, ITensor):
                all_vars.append(v)
            elif self._wrapper.is_variable(v):
                all_vars.append(self._wrapper.wrap_variable(v, name=self.name + "/" + k))
            elif isinstance(v, object):
                extra_vars.extend(self._wrapper.vars_from_object(v, self.name, k))
                for x in getattr(v, '__dict__', {}):
                    if isinstance(v.__dict__[x], StatefullObject):
                        all_vars.extend(x.variables)
                    if isinstance(v.__dict__[x], ITensor):
                        all_vars.append(x)
                    if self._wrapper.is_variable(v.__dict__[x]):
                        extra_vars.append(self._wrapper.wrap_variable(v.__dict__[x], name=self.name + "/" + x))
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

    def state_dict(self):
        state = {}
        for var in self.variables:
            state[var.name] = var.numpy()
        return state

    def load_state_dict(self, state_dict):
        for var in self.variables:
            #print("Loading: {}".format(var.name))
            var.assign(state_dict[var.name])
