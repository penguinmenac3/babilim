from typing import Union, Any, Sequence, Dict, Tuple, Optional

import numpy as np
import torch
from torch import Tensor as _Tensor
from babilim.core.logging import error
from babilim.core.itensor import ITensor, ITensorWrapper
from babilim.core.device import get_current_device_native_format


_variable_wrappers = {}


class TensorWrapper(ITensorWrapper):
    def __init__(self):
        pass

    def wrap(self, obj: Any) -> Any:
        def _wrap_indexable(obj, indices):
            obj = obj.copy()
            ret = obj
            for i in indices:
                if obj[i] is None: continue
                obj[i] = self.wrap(obj[i])
                if obj[i] is None: ret = None
            return ret
        if isinstance(obj, Tuple):
            obj = list(obj)
            obj = _wrap_indexable(obj, range(len(obj)))
            if obj is not None:
                obj = tuple(obj)
        elif isinstance(obj, Sequence):
            obj = _wrap_indexable(obj, range(len(obj)))
        elif isinstance(obj, Dict):
            obj = _wrap_indexable(obj, obj)
        elif isinstance(obj, _Tensor):
            obj = Tensor(native=obj, trainable=obj.requires_grad)
        elif isinstance(obj, np.ndarray):
            obj = Tensor(data=obj, trainable=False)
        elif isinstance(obj, Tensor):
            # Make sure that also babilim tensors are on the correct device.
            obj._auto_device()
            if obj.is_nan().any():
                error("NaN Tensor: {}".format(obj.native))
                raise ValueError("NaN Tensor {}".format(obj.native))
            obj = None
        else:
            obj = None
        return obj

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

    def wrap_variable(self, obj: Any) -> 'ITensor':
        if obj not in _variable_wrappers:
            tmp = Tensor(native=obj, trainable=obj.requires_grad)
            _variable_wrappers[obj] = tmp
        return _variable_wrappers[obj]._auto_device()

    def vars_from_object(self, v: Any, namespace: str) -> Sequence[Tuple[str, 'ITensor']]:
        extra_vars = []
        if getattr(v, 'named_buffers', False) and getattr(v, 'named_parameters', False):
            named_params = v.named_parameters()
            params = []
            for key, x in named_params:
                key = key.replace(".", "/")
                if self.is_variable(x):
                    params.append(key)
                    name = namespace + "/" + key
                    extra_vars.append((name, self.wrap_variable(x)))
            buffers = v.named_buffers()
            for key, x in buffers:
                key = key.replace(".", "/")
                if key not in params:
                    if self.is_variable(x):
                        name = namespace + "/" + key
                        extra_vars.append((name, self.wrap_variable(x)))
        return extra_vars


class Tensor(ITensor):
    def __init__(self, data: np.ndarray = None, trainable=False, native: _Tensor=None):
        if data is not None:
            #data = data.T
            native = torch.from_numpy(data)
            native.requires_grad = trainable
        elif native is not None:
            native = native
        else:
            raise RuntimeError("You must specify the data or a native value from the correct framework.")
        super().__init__(native)
        self._auto_device()

    def _auto_device(self):
        device = get_current_device_native_format()
        if str(self.native.device) != device:
            self.native = self.native.to(torch.device(get_current_device_native_format()))
        return self

    def ref(self) -> 'ITensor':
        return self.native
        
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

    def __getitem__(self, item) -> 'Tensor':
        if isinstance(item, Tensor):
            result = self.native[item.native]
        else:
            result = self.native[item]
        return Tensor(native=result)
