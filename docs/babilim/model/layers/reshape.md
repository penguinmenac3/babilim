[Back to Overview](../../../README.md)

# babilim.model.layers.reshape

> Reshape a tensor.

# *class* **Reshape**(ModuleNative)

Reshape a tensor.

A tensor of shape (B, ?) where B is the batch size gets reshaped into (B, output_shape[0], output_shape[1], ...) where the batch size is kept and all other dimensions are depending on output_shape.

* output_shape: The shape that the tensor should have after reshaping is (batch_size,) + output_shape (meaning batch size is automatically kept).


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

reshape = Reshape(output_shape=(8, 24))
tensor = Tensor(data=np.zeros((10,8,8,3), dtype=np.float32), trainable=False)

print(tensor.shape)
result = reshape(tensor)
print(result.shape)
```
Output:
```
(10, 8, 8, 3)
(10, 8, 24)

```

