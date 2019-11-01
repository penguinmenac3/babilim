from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor, TensorWrapper


_statefull_object_name_table = {}


class StatefullObject(object):
    def __init__(self, name):
        self._wrapper = TensorWrapper()
        if name not in _statefull_object_name_table:
            _statefull_object_name_table[name] = 1
            self.__name = name
        else:
            numbering = _statefull_object_name_table[name]
            self.__name = "{}_{}".format(name, numbering)
            _statefull_object_name_table[name] += 1

    @property
    def name(self):
        return self.__name

    @property
    def variables(self):
        all_vars = []
        extra_vars = []
        for member_name in self.__dict__:
            v = self.__dict__[member_name]
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
                        all_vars.append(self._wrapper.wrap_variable(x, name=self.name + "/" + member_name + "/" + k))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, self.name + "/" + member_name + "/" + k))
            elif isinstance(v, Iterable):
                for x in v:
                    if isinstance(x, StatefullObject):
                        all_vars.extend(x.variables)
                    if isinstance(x, ITensor):
                        all_vars.append(x)
                    if self._wrapper.is_variable(x):
                        all_vars.append(self._wrapper.wrap_variable(x, name=self.name + "/" + member_name + "/unnamed"))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, self.name + "/" + member_name + "/unnamed"))
            elif isinstance(v, StatefullObject):
                all_vars.extend(v.variables)
            elif isinstance(v, ITensor):
                all_vars.append(v)
            elif self._wrapper.is_variable(v):
                all_vars.append(self._wrapper.wrap_variable(v, name=self.name + "/" + member_name))
            elif isinstance(v, object):
                extra_vars.extend(self._wrapper.vars_from_object(v, self.name + "/" + member_name))
                for x in getattr(v, '__dict__', {}):
                    if isinstance(v.__dict__[x], StatefullObject):
                        all_vars.extend(v.__dict__[x].variables)
                    if isinstance(v.__dict__[x], ITensor):
                        all_vars.append(v.__dict__[x])
                    if self._wrapper.is_variable(v.__dict__[x]):
                        extra_vars.append(self._wrapper.wrap_variable(v.__dict__[x], name=self.name + "/" + member_name + "/" + x))
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
            if babilim.is_backend(babilim.TF_BACKEND):
                state[var.name] = var.numpy()
            else:
                state[var.name] = var.numpy().T
        return state

    def load_state_dict(self, state_dict):
        for var in self.variables:
            if var.name in state_dict:
                if babilim.is_backend(babilim.TF_BACKEND):
                    var.assign(state_dict[var.name])
                else:
                    var.assign(state_dict[var.name].T)
            else:
                print("Warning: Variable {} not in checkpoint.".format(var.name))
