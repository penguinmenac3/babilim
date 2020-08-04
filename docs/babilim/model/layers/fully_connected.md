[Back to Overview](../../../README.md)

# babilim.model.layers.fully_connected

> A simple fully connected layer (aka Linear Layer or Dense).

# *class* **FullyConnected**(ModuleNative)

A simple fully connected layer (aka Linear Layer or Dense).

It computes Wx+b with optional activation funciton.

* out_features: The number of output features.
* activation: The activation function that should be added after the fc layer.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

fc1 = FullyConnected(out_features=10)
tensor = Tensor(data=np.zeros((10,24), dtype=np.float32), trainable=False)

print(tensor.shape)
result = fc1(tensor)
print(result.shape)
```
Output:
```
(10, 24)
(10, 10)

```

