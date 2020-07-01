# babilim.core.itensor

> Tensor interface for babilim. Shared by tensorflow and pytorch.

This code is under the MIT License.

# *class* **ITensor**(object)

The babilim tensor api.

**You must not instantiate this class**.
Instantiate babilim.core.Tensor and use babilim.core.ITensor for your type annotations.
Every babilim.core.Tensor implements the interface type babilim.core.ITensor.

Beyond special operations every tensor implements common builtin functions like multiplication, division, addition, power etc.

* native: The native tensor that should be hidden behind this api.


### *def* **ref**(*self*) -> object

Returns a hashable reference to this tensor.

* returns: A hashable tensor.


### *def* **copy**(*self*) -> 'ITensor'

Create a copy of the tensor.

By creating a copy a new tensor which is not related to the old one (no gradients) will be created.

* returns: A copy of the tensor.


### *def* **cast**(*self*, dtype) -> 'ITensor'

Cast a tensor to a given dtype.

* dtype: The dtype as a string. Supported dtyles are "float16", "float32", "float64", "bool", "uint8", "int8", "int16", "int32" and "int64".


### *def* **stop_gradients**(*self*) -> 'ITensor'

Returns a new tensor object sharing the value with the old one but without gradients connecting them.

* returns: A new tensor handle without gradients.


### *def* **assign**(*self*, other: Union['ITensor', np.ndarray]) -> 'ITensor'

Assign the values from another tensor to this tensor.

* other: The tensor providing the values for the assignment.
* returns: Returns self for chaining of operations.


### *def* **numpy**(*self*) -> np.ndarray

Converts the tensor to a numpy array.

* returns: A numpy array with the contents of the tensor.


### *def* **mean**(*self*, axis: Optional[int]=None) -> 'ITensor'

Computes the mean operation on the tensor.

* axis: (Optional) An axis along which the mean should be computed. If none is given, all axis are reduced.
* returns: A tensor containing the mean.


### *def* **argmax**(*self*, axis: Optional[int]=None) -> 'ITensor'

Computes the argmax operation on the tensor.

* axis: (Optional) An axis along which the argmax should be computed. If none is given, all axis are reduced.
* returns: A tensor containing the argmax.


### *def* **sum**(*self*, axis: Optional[int]=None) -> 'ITensor'

Computes the sum operation on the tensor.

* axis: (Optional) An axis along which the sum should be computed. If none is given, all axis are reduced.
* returns: A tensor containing the sum.


### *def* **is_nan**(*self*) -> 'ITensor'

Checks for each value in the tensor if it is not a number.

* returns: A tensor containing booleans if the value in the original tensors were nan.


### *def* **any**(*self*) -> bool

Check if any vlaue of the tensor is true.

* returns: True if any value was true.


### *def* **shape**(*self*) -> Tuple

Get the shape of the tensor.

* returns: A tuple representing the shape of the tensor.


### *def* **trainable**(*self*) -> bool

Check if a tensor is trainable.

Essentially this is used to find out if a tensor should be updated during back propagation.

* returns: True if the tensor can be trained.


# Interoperability between Pytorch/Tensorflow and Babilim

Sometimes it is nescesarry to implement stuff in native pytorch or native tensorflow. Here the tensor wrapper can help.

**WARNING: Instead of directly using the TensorWrapper, you should prefer using the babilim.module.Lambda!**

# *class* **ITensorWrapper**(object)

This interface implements functions required to wrap variables from native pytorch/tensorflow code for babilim.


### *def* **wrap**(*self*, obj: Any) -> Optional[Any]

Wrap simple datatypes into a tensor.

Supported datatypes are Tuple, Sequence, Dict, native tensors and numpy arrays.

For Tuple, Dict, Sequence it wraps every element of the object recursively.
For native tensors and numpy arrays it wraps them as an ITensor.

* obj: The object that should be wrapped.
* returns: The wrapped object or none if object cannot be wrapped.


### *def* **unwrap**(*self*, obj: Any) -> Any

Unwrap an object.

This function is the inverse to wrap (if the object was wrapped).

* returns: The unwrapped object again.


### *def* **is_variable**(*self*, obj: Any) -> bool

Check if an object is a variable in the framework.

* obj: The object that should be tested.
* returns: True if the object is a tensorflow/pytorch variable.


### *def* **wrap_variable**(*self*, obj: Any) -> 'ITensor'

Wrap a variable as an ITensor.

* obj: The object that should be wrapped.
* returns: An ITensor object containing the variable.


### *def* **vars_from_object**(*self*, obj: Any, namespace: str) -> Sequence[Tuple[str, 'ITensor']]

Get all variables in an native module.

This function retrieves all variables from a native module and converts them in a list of mappings from name to ITensor.

* obj: The native module from which to extract the variables.
* namespace: The namespace that should be used for the variables found.
* returns: A list containing tuples mapping from a name (str) to the ITensor.


