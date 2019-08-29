from typing import Union, Any, Sequence, Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor as _Tensor
from babilim.core.itensor import ITensor, ITensorWrapper


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
        if isinstance(obj, Dict):
            for k in obj:
                obj[k], q = self.wrap(obj[k])
                wrapped = wrapped or q
        if isinstance(obj, tf.Variable):
            obj = Tensor(native=obj, trainable=obj.trainable)
            wrapped = True
        elif isinstance(obj, _Tensor):
            obj = Tensor(native=obj)
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


class Tensor(ITensor):
    def __init__(self, data: np.ndarray = None, trainable: bool = False, native=None, order_flipped: bool = False):
        self.order_flipped = order_flipped
        if data is not None:
            if self.order_flipped:
                data = data.T
            native = tf.Variable(data, trainable=trainable)
        elif native is not None:
            native = native
        else:
            raise RuntimeError("You must specify the data or a native value from the correct framework.")
        super().__init__(native)

    def assign(self, other: Union['Tensor', np.ndarray]) -> 'Tensor':
        if isinstance(other, np.ndarray):
            if self.order_flipped:
                other = other.T
            self.assign(Tensor(data=other, trainable=self.trainable))
        elif isinstance(self.native, tf.Variable):
            self.native.assign(other.native)
        else:
            self.native = other.native
        return self

    def numpy(self) -> np.ndarray:
        if self.order_flipped:
            return self.native.numpy().T
        return self.native.numpy()

    @property
    def shape(self) -> Tuple:
        return self.native.shape

    @property
    def trainable(self) -> bool:
        return self.native.trainable

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
