[Back to Overview](../../../README.md)

# babilim.model.layers.tensor_combiners

> Ways of combining tensors.

# *class* **Stack**(ModuleNative)

Stack layers along an axis.

Creates a callable object with the following signature:
* **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
* **return**: A tensor of shape [..., S, ...] where the position at which S is in the shape is equal to the axis.

Parameters of the constructor.
* axis: (int) The axis along which the stacking happens.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

stack = Stack(axis=1)
tensor1 = Tensor(data=np.zeros((10,8,8,3)), trainable=False)
tensor2 = Tensor(data=np.zeros((10,8,8,3)), trainable=False)

print(tensor1.shape)
print(tensor2.shape)
result = stack([tensor1, tensor2])
print(result.shape)
```
Output:
```
(10, 8, 8, 3)
(10, 8, 8, 3)
(10, 2, 8, 8, 3)

```

# *class* **Concat**(ModuleNative)

Concatenate layers along an axis.

Creates a callable object with the following signature:
* **tensor_list**: (List[Tensor]) The tensors that should be stacked. A list of length S containing Tensors.
* **return**: A tensor of shape [..., S * inp_tensor.shape[axis], ...] where the position at which S is in the shape is equal to the axis.

Parameters of the constructor.
* axis: (int) The axis along which the concatenation happens.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

stack = Concat(axis=1)
tensor1 = Tensor(data=np.zeros((10,8,8,3)), trainable=False)
tensor2 = Tensor(data=np.zeros((10,8,8,3)), trainable=False)

print(tensor1.shape)
print(tensor2.shape)
result = stack([tensor1, tensor2])
print(result.shape)
```
Output:
```
(10, 8, 8, 3)
(10, 8, 8, 3)
(10, 16, 8, 3)

```

