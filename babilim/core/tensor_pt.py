from typing import Union, Any, Sequence, Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor as _Tensor
from babilim.core.itensor import ITensor, ITensorWrapper


_variable_wrappers = {}


class TensorWrapper(ITensorWrapper):
    def wrap(self, obj: Any) -> Any:
        wrapped = False
        if isinstance(obj, Tuple):
            obj = list(obj)
            for i in range(len(obj)):
                obj[i], q = self.wrap(obj[i])
                wrapped = wrapped or q
            obj = tuple(obj)
        elif isinstance(obj, Sequence):
            for i in range(len(obj)):
                obj[i], q = self.wrap(obj[i])
                wrapped = wrapped or q
        elif isinstance(obj, Dict):
            for k in obj:
                obj[k], q = self.wrap(obj[k])
                wrapped = wrapped or q
        elif isinstance(obj, _Tensor):
            obj = Tensor(native=obj, trainable=obj.requires_grad)
            # FIXME is this required? obj should already be cuda
            if not obj.native.is_cuda and torch.cuda.is_available():
                obj.native = obj.native.to(torch.device("cuda"))
            wrapped = True
        elif isinstance(obj, np.ndarray):
            obj = Tensor(data=obj, trainable=False)
            wrapped = True
        return obj, wrapped
    
    def unwrap(self, obj: Any) -> Any:
        if isinstance(obj, Sequence):
            for i in range(len(obj)):
                obj[i] = self.unwrap(obj[i])
        if isinstance(obj, Dict):
            for k in obj:
                obj[k] = self.unwrap(obj[k])
        if isinstance(obj, Tensor):
            obj = obj.native
        return obj

    def is_variable(self, obj: Any) -> bool:
        return isinstance(obj, _Tensor)

    def wrap_variable(self, obj: Any, name: str) -> 'ITensor':
        # TODO remove name from API
        if obj not in _variable_wrappers:
            tmp = Tensor(native=obj, trainable=obj.requires_grad)
            if not tmp.native.is_cuda and torch.cuda.is_available():
                tmp.native = tmp.native.to(torch.device("cuda"))
            _variable_wrappers[obj] = tmp
        return _variable_wrappers[obj]

    def vars_from_object(self, v: Any, namespace: str) -> Sequence[Tuple[str, 'ITensor']]:
        extra_vars = []
        if getattr(v, 'state_dict', False) and getattr(v, 'named_parameters', False):
            named_params = v.named_parameters()
            params = []
            for key, x in named_params:
                key = key.replace(".", "/")
                if self.is_variable(x):
                    params.append(key)
                    name = namespace + "/" + key
                    extra_vars.append((name, self.wrap_variable(x, name=name)))
            state = v.state_dict()
            for key, x in state.items():
                key = key.replace(".", "/")
                if key not in params:
                    if self.is_variable(x):
                        name = namespace + "/" + key
                        extra_vars.append((name, self.wrap_variable(x, name=name)))
        return extra_vars


class Tensor(ITensor):
    def __init__(self, data: np.ndarray = None, trainable=False, native: _Tensor=None):
        if data is not None:
            #data = data.T
            native = torch.from_numpy(data)
            native.requires_grad = trainable
            if torch.cuda.is_available() and not native.is_cuda:
                native = native.to(torch.device("cuda"))
        elif native is not None:
            native = native
        else:
            raise RuntimeError("You must specify the data or a native value from the correct framework.")
        super().__init__(native)

    def copy(self) -> 'Tensor':
        return Tensor(data=self.numpy(), trainable=self.trainable)
        
    def cast(self, dtype) -> 'ITensor':
        if dtype == "float16":
            return Tensor(native=self.native.half())
        elif dtype == "float32":
            return Tensor(native=self.native.float())
        elif dtype == "float64":
            return Tensor(native=self.native.double())
        elif dtype == "bool":
            return Tensor(native=self.native.bool())
        elif dtype == "uint8":
            return Tensor(native=self.native.byte())
        elif dtype == "int8":
            return Tensor(native=self.native.char())
        elif dtype == "int16":
            return Tensor(native=self.native.short())
        elif dtype == "int32":
            return Tensor(native=self.native.int())
        elif dtype == "int64":
            return Tensor(native=self.native.long())
        else:
            raise RuntimeError("dtype {} not valid".format(dtype))

    def stop_gradients(self) -> 'Tensor':
        return Tensor(native=self.native.detach())

    def assign(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        if isinstance(other, np.ndarray):
            #other = other.T
            self.assign(Tensor(data=other, trainable=self.trainable))
        else:
            self.native.data = other.native
        return self

    def numpy(self) -> np.ndarray:
        tmp = self.native
        if tmp.requires_grad:
            tmp = tmp.detach()
        if tmp.is_cuda:
            tmp = tmp.cpu()
        
        return tmp.numpy()

    def mean(self, axis: Optional[int]=None) -> 'Tensor':
        if axis is not None:
            return Tensor(native=self.native.mean(dim=axis))
        else:
            return Tensor(native=self.native.mean())
    
    def argmax(self, axis: Optional[int]=None) -> 'ITensor':
        if axis is not None:
            return Tensor(native=self.native.argmax(dim=axis))
        else:
            return Tensor(native=self.native.argmax())

    def sum(self, axis: Optional[int]=None) -> 'ITensor':
        if axis is not None:
            return Tensor(native=self.native.sum(dim=axis))
        else:
            return Tensor(native=self.native.sum())

    def is_nan(self) -> 'ITensor':
        return Tensor(native=torch.isnan(self.native))

    def any(self) -> bool:
        return self.native.any()

    @property
    def shape(self) -> Tuple:
        return tuple(self.native.shape)

    @property
    def trainable(self) -> bool:
        return self.native.requires_grad

    def __str__(self):
        return str(self.native)

    def __repr__(self):
        return repr(self.native)

    # Binary Operators
    def __add__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native + other.native)
        else:
            return Tensor(native=self.native + other)

    def __sub__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native - other.native)
        else:
            return Tensor(native=self.native - other)
    
    def __mul__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native * other.native)
        else:
            return Tensor(native=self.native * other)

    def __truediv__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native / other.native)
        else:
            return Tensor(native=self.native / other)

    def __mod__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native % other.native)
        else:
            return Tensor(native=self.native % other)

    def __pow__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native ** other.native)
        else:
            return Tensor(native=self.native ** other)

    # Comparison Operators
    def __lt__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native < other.native)

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native > other.native)

    def __le__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native <= other.native)

    def __ge__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native >= other.native)

    def __eq__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native == other.native)

    def __ne__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(native=self.native != other.native)

    # Unary Operators
    def __neg__(self) -> 'Tensor':
        return Tensor(native=-self.native)

    def __pos__(self) -> 'Tensor':
        return self

    def __invert__(self) -> 'Tensor':
        return Tensor(native=~self.native)
