from typing import Sequence, Any, Sequence, Callable, Dict, Iterable
from collections import defaultdict
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND, info, warn, DEBUG_VERBOSITY
from babilim.core.checkpoint import Checkpoint
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor, TensorWrapper


_statefull_object_name_table = {}


class StatefullObject(object):
    def __init__(self):
        self._wrapper = TensorWrapper()
        self._training = True

    @property
    def training(self):
        return self._training

    @property
    def variables(self):
        return list(self.named_variables.values())

    @property
    def named_variables(self):
        return dict(self.__variables_with_namespace())

    def __variables_with_namespace(self, namespace=""):
        all_vars = []
        extra_vars = []
        for member_name in self.__dict__:
            v = self.__dict__[member_name]
            if isinstance(v, str):
                pass
            elif isinstance(v, Dict):
                for i, (k, x) in enumerate(v.items()):
                    if not isinstance(k, str):
                        k = "{}".format(i)
                    name = namespace + "/" + member_name + "/" + k
                    if isinstance(x, StatefullObject):
                        all_vars.extend(x.__variables_with_namespace(name))
                    if isinstance(x, ITensor):
                        all_vars.append((name, x))
                    if self._wrapper.is_variable(x):
                        all_vars.append((name, self._wrapper.wrap_variable(x, name=name)))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, name))
            elif isinstance(v, Iterable):
                for i, x in enumerate(v):
                    name = namespace + "/" + member_name + "/{}".format(i)
                    if isinstance(x, StatefullObject):
                        all_vars.extend(x.__variables_with_namespace(name))
                    if isinstance(x, ITensor):
                        all_vars.append((name, x))
                    if self._wrapper.is_variable(x):
                        all_vars.append((name, self._wrapper.wrap_variable(x, name=name)))
                    if isinstance(x, object):
                        extra_vars.extend(self._wrapper.vars_from_object(v, name))
            elif isinstance(v, StatefullObject):
                name = namespace + "/" + member_name
                all_vars.extend(v.__variables_with_namespace(name))
            elif isinstance(v, ITensor):
                name = namespace + "/" + member_name
                all_vars.append((name, v))
            elif self._wrapper.is_variable(v):
                name = namespace + "/" + member_name
                all_vars.append((name, self._wrapper.wrap_variable(v, name=name)))
            elif isinstance(v, object):
                name = namespace + "/" + member_name
                extra_vars.extend(self._wrapper.vars_from_object(v, name))
                for x in getattr(v, '__dict__', {}):
                    name = namespace + "/" + member_name + "/" + x
                    if isinstance(v.__dict__[x], StatefullObject):
                        all_vars.extend(v.__dict__[x].__variables_with_namespace(name))
                    if isinstance(v.__dict__[x], ITensor):
                        all_vars.append((name, v.__dict__[x]))
                    if self._wrapper.is_variable(v.__dict__[x]):
                        extra_vars.append((name, self._wrapper.wrap_variable(v.__dict__[x], name=name)))
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
    def named_trainable_variables(self):
        all_vars = self.named_variables
        train_vars = []
        for k, v in all_vars.items():
            if v.trainable:
                train_vars.append((k, v))
        return dict(train_vars)

    @property
    def untrainable_variables(self):
        all_vars = self.variables
        train_vars = []
        for v in all_vars:
            if not v.trainable:
                train_vars.append(v)
        return train_vars

    @property
    def named_untrainable_variables(self):
        all_vars = self.named_variables
        train_vars = []
        for k, v in all_vars.items():
            if not v.trainable:
                train_vars.append((k, v))
        return dict(train_vars)

    @property
    def trainable_variables_native(self):
        all_vars = self.trainable_variables
        train_vars = []
        for v in all_vars:
            train_vars.append(v.native)
        return train_vars

    def state_dict(self):
        state = {}
        for name, var in self.named_variables.items():
            if babilim.is_backend(babilim.TF_BACKEND):
                state[name] = var.numpy()
            else:
                state[name] = var.numpy().T
        return state

    def load_state_dict(self, state_dict):
        for name, var in self.named_variables.items():
            if name in state_dict:
                if babilim.is_backend(babilim.TF_BACKEND):
                    var.assign(state_dict[name])
                else:
                    var.assign(state_dict[name].T)
                if DEBUG_VERBOSITY:
                    info("  Loaded: {}".format(name))
            else:
                warn("  Variable {} not in checkpoint.".format(name))

    def eval(self):
        self.train(False)

    def train(self, mode=True):
        self._training = mode
        for member_name in self.__dict__:
            obj = self.__dict__[member_name]
            if isinstance(obj, Sequence):
                for x in obj:
                    train_fn = getattr(x, "train", None)
                    if callable(train_fn):
                        train_fn(mode)
            else:
                train_fn = getattr(obj, "train", None)
                if callable(train_fn):
                    train_fn(mode)

    def load(self, checkpoint_file_path):
        checkpoint = Checkpoint(checkpoint_file_path)
        if DEBUG_VERBOSITY:
            checkpoint.print()
        model_state = checkpoint.get_state_dict("model")
        if len(model_state) > 0:
            self.load_state_dict(model_state)
        else:
            babilim.error("Could not find state in checkpoint.")

    def save(self, checkpoint_file_path):
        checkpoint = Checkpoint(checkpoint_file_path)
        checkpoint.set_state_dict("model", self.state_dict())
        if DEBUG_VERBOSITY:
            checkpoint.print()
        checkpoint.save()
