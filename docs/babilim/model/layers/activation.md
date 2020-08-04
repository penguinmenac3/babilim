[Back to Overview](../../../README.md)

# babilim.model.layers.activation

> Compute an activation function.

# *class* **Activation**(ModuleNative)

Supports the activation functions.

* activation: A string specifying the activation function to use. (Only "relu" and None supported yet.)


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

activation = Activation(activation="relu")
tensor = Tensor(data=np.array([-1.0, -0.5, 0, 0.5, 1.0], dtype=np.float32), trainable=False)

print(tensor.numpy())
result = activation(tensor)
print(result.numpy())
```
Output:
```
[-1.  -0.5  0.   0.5  1. ]
[0.  0.  0.  0.5 1. ]

```

