[Back to Overview](../../../README.md)

# babilim.model.layers.flatten

> Flatten a feature map into a linearized tensor.

# *class* **Flatten**(ModuleNative)

Flatten a feature map into a linearized tensor.

This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

flatten = Flatten()
tensor = Tensor(data=np.zeros((10,8,8,3)), trainable=False)

print(tensor.shape)
result = flatten(tensor)
print(result.shape)
```
Output:
```
(10, 8, 8, 3)
(10, 192)

```

