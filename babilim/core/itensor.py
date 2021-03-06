# AUTOGENERATED FROM: babilim/core/itensor.ipynb

# Cell: 0
"""doc
# babilim.core.itensor

> Tensor interface for babilim. Shared by tensorflow and pytorch.

This code is under the MIT License.
"""

# Cell: 1
# MIT License
#
# Copyright (c) 2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Union, Any, Tuple, Optional, Sequence
import numpy as np

tensor_name_table = {}


# Cell: 2
class ITensor(object):
    def __init__(self, native):
        """
        The babilim tensor api.
        
        **You must not instantiate this class**.
        Instantiate babilim.core.Tensor and use babilim.core.ITensor for your type annotations.
        Every babilim.core.Tensor implements the interface type babilim.core.ITensor.
        
        Beyond special operations every tensor implements common builtin functions like multiplication, division, addition, power etc.
        
        :param native: The native tensor that should be hidden behind this api.
        """
        self.native = native
    
    def ref(self) -> object:
        """
        Returns a hashable reference to this tensor.
        
        :return: A hashable tensor.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def copy(self) -> 'ITensor':
        """
        Create a copy of the tensor.
        
        By creating a copy a new tensor which is not related to the old one (no gradients) will be created.
        
        :return: A copy of the tensor.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def cast(self, dtype) -> 'ITensor':
        """
        Cast a tensor to a given dtype.
        
        :param dtype: The dtype as a string. Supported dtyles are "float16", "float32", "float64", "bool", "uint8", "int8", "int16", "int32" and "int64".
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def stop_gradients(self) -> 'ITensor':
        """
        Returns a new tensor object sharing the value with the old one but without gradients connecting them.
        
        :return: A new tensor handle without gradients.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def assign(self, other: Union['ITensor', np.ndarray]) -> 'ITensor':
        """
        Assign the values from another tensor to this tensor.
        
        :param other: The tensor providing the values for the assignment.
        :return: Returns self for chaining of operations.
        """
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def reshape(self, shape) -> 'ITensor':
        """
        Reshape a tensor into a given shape.

        :param shape: (Tuple) The desired target shape.
        :return: (ITensor) Returns a view on the tensor with the new shape.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def transpose(self, axis_a=0, axis_b=1) -> 'ITensor':
        """
        Transpose a tensor by swapping two axis.

        :param axis_a: (Optional[int]) The axis that should be swapped. Default: 0
        :param axis_b: (Optional[int]) The axis that should be swapped. Default: 1
        :return: (ITensor) Returns a tensor where axis_a and axis_b are swapped.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def numpy(self) -> np.ndarray:
        """
        Converts the tensor to a numpy array.
        
        :return: A numpy array with the contents of the tensor.
        """
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def mean(self, axis: Optional[int]=None) -> 'ITensor':
        """
        Computes the mean operation on the tensor.
        
        :param axis: (Optional) An axis along which the mean should be computed. If none is given, all axis are reduced.
        :return: A tensor containing the mean.
        """
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def min(self, axis: Optional[int]=None) -> 'ITensor':
        """
        Computes the min operation on the tensor.
        
        :param axis: (Optional) An axis along which the min should be computed. If none is given, all axis are reduced.
        :return: A tensor containing the min.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def max(self, axis: Optional[int]=None) -> 'ITensor':
        """
        Computes the max operation on the tensor.
        
        :param axis: (Optional) An axis along which the max should be computed. If none is given, all axis are reduced.
        :return: A tensor containing the max.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def argmax(self, axis: Optional[int]=None) -> 'ITensor':
        """
        Computes the argmax operation on the tensor.
        
        :param axis: (Optional) An axis along which the argmax should be computed. If none is given, all axis are reduced.
        :return: A tensor containing the argmax.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def sum(self, axis: Optional[int]=None) -> 'ITensor':
        """
        Computes the sum operation on the tensor.
        
        :param axis: (Optional) An axis along which the sum should be computed. If none is given, all axis are reduced.
        :return: A tensor containing the sum.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def is_nan(self) -> 'ITensor':
        """
        Checks for each value in the tensor if it is not a number.
        
        :return: A tensor containing booleans if the value in the original tensors were nan.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def any(self) -> bool:
        """
        Check if any vlaue of the tensor is true.
        
        :return: True if any value was true.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def all(self) -> bool:
        """
        Check if all vlaues of the tensor are true.
        
        :return: True if all values are true.
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    def repeat(self, repeats, axis) -> 'ITensor':
        """
        Repeat a tensor along an axis for n times.
        
        :param repeats: The n how often the tensor should be repeated.
        :param axis: The axis along which to repeat.
        :return: A copy of the tensor which is repeated n times along the axis
        """
        raise NotImplementedError("Each implementation of a tensor must implement this.")

    @property
    def shape(self) -> Tuple:
        """
        Get the shape of the tensor.
        
        :return: A tuple representing the shape of the tensor.
        """
        raise NotImplementedError("Each implementation of a variable must implement this.")

    @property
    def trainable(self) -> bool:
        """
        Check if a tensor is trainable.
        
        Essentially this is used to find out if a tensor should be updated during back propagation.
        
        :return: True if the tensor can be trained.
        """
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

    def __floordiv__(self, other: Union[int, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __mod__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __pow__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    # Comparison Operators
    def __lt__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __gt__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __le__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __ge__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __eq__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __ne__(self, other: Union[float, 'ITensor']) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    # Unary Operators
    def __neg__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __pos__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __invert__(self) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __getitem__(self, item) -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")

    def __setitem__(self, item, value) -> None:
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

    def __and__(self, other: 'ITensor') -> 'ITensor':
        raise NotImplementedError("Each implementation of a variable must implement this.")


# Cell: 3
"""doc
# Interoperability between Pytorch/Tensorflow and Babilim

Sometimes it is nescesarry to implement stuff in native pytorch or native tensorflow. Here the tensor wrapper can help.

**WARNING: Instead of directly using the TensorWrapper, you should prefer using the babilim.module.Lambda!**
"""

# Cell: 4
class ITensorWrapper(object):
    def __init__(self):
        """
        This interface implements functions required to wrap variables from native pytorch/tensorflow code for babilim.
        """
        raise RuntimeError("This interface must not be instantiated!")
    
    def wrap(self, obj: Any) -> Optional[Any]:
        """
        Wrap simple datatypes into a tensor.
        
        Supported datatypes are Tuple, Sequence, Dict, native tensors and numpy arrays.
        
        For Tuple, Dict, Sequence it wraps every element of the object recursively.
        For native tensors and numpy arrays it wraps them as an ITensor.
        
        :param obj: The object that should be wrapped.
        :return: The wrapped object or none if object cannot be wrapped.
        """
        raise NotImplementedError()

    def unwrap(self, obj: Any) -> Any:
        """
        Unwrap an object.
        
        This function is the inverse to wrap (if the object was wrapped).
        
        :return: The unwrapped object again.
        """
        raise NotImplementedError()

    def is_variable(self, obj: Any) -> bool:
        """
        Check if an object is a variable in the framework.
        
        :param obj: The object that should be tested.
        :return: True if the object is a tensorflow/pytorch variable.
        """
        raise NotImplementedError()

    def wrap_variable(self, obj: Any) -> 'ITensor':
        """
        Wrap a variable as an ITensor.
        
        :param obj: The object that should be wrapped.
        :return: An ITensor object containing the variable.
        """
        raise NotImplementedError()

    def vars_from_object(self, obj: Any, namespace: str) -> Sequence[Tuple[str, 'ITensor']]:
        """
        Get all variables in an native module.
        
        This function retrieves all variables from a native module and converts them in a list of mappings from name to ITensor.
        
        :param obj: The native module from which to extract the variables.
        :param namespace: The namespace that should be used for the variables found.
        :return: A list containing tuples mapping from a name (str) to the ITensor.
        """
        raise NotImplementedError()
