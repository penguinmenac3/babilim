from typing import Union, Any, Sequence, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import Tensor as _Tensor
from babilim.core.itensor import ITensor, ITensorWrapper


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
        elif isinstance(obj, tf.Variable):
            obj = Tensor(native=obj, trainable=obj.trainable)
        elif isinstance(obj, _Tensor):
            obj = Tensor(native=obj)
        elif isinstance(obj, np.ndarray):
            obj = Tensor(data=obj, trainable=False)
        else:
            obj = None
        return obj
    
    def unwrap(self, obj: Any) -> Any:
        def _unwrap_indexable(obj, indices):
            obj = obj.copy()
            ret = obj
            for i in indices:
                if obj[i] is None: continue
                obj[i] = self.unwrap(obj[i])
                if obj[i] is None: ret = None
            return ret
        if isinstance(obj, Tuple):
            obj = list(obj)
            obj = _unwrap_indexable(obj, range(len(obj)))
            if obj is not None:
                obj = tuple(obj)
        elif isinstance(obj, Sequence):
            obj = _unwrap_indexable(obj, range(len(obj)))
        elif isinstance(obj, Dict):
            obj = _unwrap_indexable(obj, obj)
        elif isinstance(obj, Tensor):
            obj = obj.native
        else:
            obj = None
        return obj

    def is_variable(self, obj: Any) -> bool:
        return isinstance(obj, tf.Variable)

    def wrap_variable(self, obj: Any) -> 'ITensor':
        if obj.ref() not in _variable_wrappers:
            _variable_wrappers[obj.ref()] = Tensor(native=obj, trainable=obj.trainable)
        return _variable_wrappers[obj.ref()]

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

    def ref(self) -> 'ITensor':
        return self.native.ref()
        
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

    def reshape(self, shape) -> 'Tensor':
        return Tensor(native=tf.reshape(self.native, shape))

    def transpose(self, axis_a=0, axis_b=1) -> 'Tensor':
        permutation = [i for i in range(len(self.shape))]
        permutation[axis_a] = axis_b
        permutation[axis_b] = axis_a
        return Tensor(native=tf.transpose(self.native, perm=permutation))

    def numpy(self) -> np.ndarray:
        return self.native.numpy()

    def mean(self, axis: Optional[int]=None) -> 'Tensor':
        return Tensor(native=tf.reduce_mean(self.native, axis=axis))

    def min(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.min(self.native, axis=axis))

    def max(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.max(self.native, axis=axis))

    def argmax(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.argmax(self.native, axis=axis))

    def sum(self, axis: Optional[int]=None) -> 'ITensor':
        return Tensor(native=tf.reduce_sum(self.native, axis=axis))

    def is_nan(self) -> 'ITensor':
        return Tensor(native=tf.math.is_nan(self.native))

    def any(self) -> bool:
        return tf.reduce_any(self.native)

    def all(self) -> bool:
        return tf.reduce_all(self.native)

    @property
    def shape(self) -> Tuple:
        return self.native.shape

    @property
    def trainable(self) -> bool:
        return True  # FIXME figure out how self.native.trainable works now

    def __str__(self):
        return str(self.native)

    def __repr__(self):
        return "Tensor({})".format(repr(self.native))

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

    def __floordiv__(self, other: Union[int, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native // other.native)
        else:
            return Tensor(native=self.native // other)

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
    def __lt__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native < other.native)
        else:
            return Tensor(native=self.native < other)

    def __gt__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native > other.native)
        else:
            return Tensor(native=self.native > other)

    def __le__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native <= other.native)
        else:
            return Tensor(native=self.native <= other)

    def __ge__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native >= other.native)
        else:
            return Tensor(native=self.native >= other)

    def __eq__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native == other.native)
        else:
            return Tensor(native=self.native == other)

    def __ne__(self, other: Union[float, 'Tensor']) -> 'Tensor':
        if isinstance(other, Tensor):
            return Tensor(native=self.native != other.native)
        else:
            return Tensor(native=self.native != other)

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

    def __setitem__(self, item, value) -> None:
        if isinstance(value, Tensor):
            value = value.native
        if isinstance(item, Tensor):
            self.native[item.native] = value.native
        else:
            self.native[item] = value.native

    def __and__(self, other: 'ITensor') -> 'ITensor':
        return Tensor(native=self.native & other.native)
