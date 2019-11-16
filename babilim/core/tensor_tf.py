from typing import Union, Any, Sequence, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import Tensor as _Tensor
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
        elif isinstance(obj, tf.Variable):
            obj = Tensor(native=obj, trainable=obj.trainable)
            wrapped = True
        elif isinstance(obj, _Tensor):
            obj = Tensor(native=obj)
            wrapped = True
        elif isinstance(obj, np.ndarray):
            obj = Tensor(data=obj, trainable=False)
            wrapped = True
        return obj, wrapped
    
    def unwrap(self, obj: Any) -> Any:
        out = None
        if isinstance(obj, Sequence):
            out = []
            for i in range(len(obj)):
                out.append(self.unwrap(obj[i]))
        if isinstance(obj, Dict):
            out = {}
            for k in obj:
                out[k] = self.unwrap(obj[k])
        if isinstance(obj, Tensor):
            out = obj.native
        return out

    def is_variable(self, obj: Any) -> bool:
        return isinstance(obj, tf.Variable)

    def wrap_variable(self, obj: Any, name: str) -> 'ITensor':
        if obj not in _variable_wrappers:
            _variable_wrappers[obj] = Tensor(native=obj, trainable=obj.trainable, name=name)
        return _variable_wrappers[obj]

    def vars_from_object(self, v: Any, namespace: str) -> Sequence[Tuple[str, 'ITensor']]:
        extra_vars = []
        # TODO is there something special to tensorflow or keras modules?
        if getattr(v, '_parameters', False):
            for x in getattr(v, '_parameters'):
                if self.is_variable(v._parameters[x]):
                    name = namespace + "/" + x
                    extra_vars.append((name, self.wrap_variable(v._parameters[x], name=name)))
        elif getattr(v, 'parameters', False):
            for x in getattr(v, 'parameters')():
                if self.is_variable(x):
                    name = namespace + "/unnamed"  # FIXME
                    extra_vars.append((name, self.wrap_variable(x, name=name)))
        return extra_vars


class Tensor(ITensor):
    def __init__(self, data: np.ndarray = None, trainable: bool = False, native=None):
        if data is not None:
            native = tf.Variable(data, trainable=trainable)
        elif native is not None:
            native = native
        else:
            raise RuntimeError("You must specify the data or a native value from the correct framework.")
        super().__init__(native)

    def copy(self) -> 'Tensor':
        return Tensor(data=self.numpy(), trainable=self.trainable)
        
    def cast(self, dtype) -> 'Tensor':
        return Tensor(native=tf.cast(self.native, dtype))

    def stop_gradients(self) -> 'Tensor':
        return Tensor(native=tf.stop_gradient(self.native))

    def assign(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        if isinstance(other, np.ndarray):
            self.assign(Tensor(data=other, trainable=self.trainable))
        elif isinstance(self.native, tf.Variable):
            self.native.assign(other.native)
        else:
            self.native = other.native
        return self

    def numpy(self) -> np.ndarray:
        return self.native.numpy()

    def mean(self, axis: Optional[int]=None) -> 'Tensor':
        return Tensor(native=tf.reduce_mean(self.native, axis=axis))
        
    def argmax(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.argmax(self.native, axis=axis))

    def sum(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.reduce_sum(self.native, axis=axis))

    def is_nan(self) -> 'ITensor':
        return Tensor(native=tf.math.is_nan(self.native))

    def any(self) -> bool:
        return tf.reduce_any(self.native)

    @property
    def shape(self) -> Tuple:
        return self.native.shape

    @property
    def trainable(self) -> bool:
        return True  # FIXME figure out how self.native.trainable works now

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
