from typing import Union, Any, Tuple
import numpy as np

tensor_name_table = {}

class ITensorWrapper(object):
    def wrap(self, obj: Any) -> Any:
        raise NotImplementedError()
    
    def unwrap(self, obj: Any) -> Any:
        raise NotImplementedError()


class ITensor(object):
    def __init__(self, native, name: str = "unnamed"):
        self.native = native
        if name not in tensor_name_table:
            tensor_name_table[name] = 0
        numbering = tensor_name_table[name]
        self.name = "{}_{}".format(name, numbering)
        tensor_name_table[name] += 1

    def copy(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a tensor must implement this.")
    
    def cast(self, dtype) -> 'ITensor':
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def stop_gradients(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def assign(self, other: Union['ITensor', np.ndarray]) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")
        
    def numpy(self) -> np.ndarray:
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def mean(self, axis: int=None) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")
    
    def argmax(self, axis: int=None) -> 'ITensor':
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    @property
    def shape(self) -> Tuple:
        raise NotImplementedError("Each implementation of a variable must implement this.")

    @property
    def trainable(self) -> bool:
        raise NotImplementedError("Each implementation of a variable must implement this.")

    # Binary Operators
    def __add__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __sub__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")
    
    def __mul__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __truediv__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __mod__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __pow__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    # Comparison Operators
    def __lt__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __gt__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __le__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __ge__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __eq__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __ne__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    # Unary Operators
    def __neg__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __pos__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __invert__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")


    # Assignment Operators
    def __isub__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self - other)

    def __iadd__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self + other)

    def __imul__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self * other)

    def __idiv__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self / other)

    def __imod__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self % other)

    def __ipow__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        return self.assign(self ** other)
