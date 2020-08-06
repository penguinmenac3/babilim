[Back to Overview](../../../README.md)

# babilim.model.layers.pooling

> Pooling operations.

---
---
## *class* **MaxPooling1D**(ModuleNative)

A N max pooling layer.

Computes the max of a N region with stride S.
This divides the feature map size by S.

* **pool_size**: Size of the region over which is pooled.
* **stride**: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

max_pool_1d = MaxPooling1D()
tensor = Tensor(data=np.zeros((10,8,16)), trainable=False)

print(tensor.shape)
result = max_pool_1d(tensor)
print(result.shape)
```
Output:
```
(10, 8, 16)
(10, 8, 8)

```

---
---
## *class* **MaxPooling2D**(ModuleNative)

A NxN max pooling layer.

Computes the max of a NxN region with stride S.
This divides the feature map size by S.

* **pool_size**: Size of the region over which is pooled.
* **stride**: The stride defines how the top left corner of the pooling moves across the image. If None then it is same to pool_size resulting in zero overlap between pooled regions.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

max_pool_2d = MaxPooling2D()
tensor = Tensor(data=np.zeros((10,8,16,32)), trainable=False)

print(tensor.shape)
result = max_pool_2d(tensor)
print(result.shape)
```
Output:
```
(10, 8, 16, 32)
(10, 8, 8, 16)

```

---
---
## *class* **GlobalAveragePooling1D**(ModuleNative)

A global average pooling layer.

This computes the global average in N dimension (B, N, C), so that the result is of shape (B, C).


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

global_avg_pool_1d = GlobalAveragePooling1D()
tensor = Tensor(data=np.zeros((10,8,5)), trainable=False)

print(tensor.shape)
result = global_avg_pool_1d(tensor)
print(result.shape)
```
Output:
```
(10, 8, 5)
(10, 8)

```

---
---
## *class* **GlobalAveragePooling2D**(ModuleNative)

A global average pooling layer.

This computes the global average in W, H dimension, so that the result is of shape (B, C).


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

global_avg_pool_2d = GlobalAveragePooling2D()
tensor = Tensor(data=np.zeros((10,8,5,3)), trainable=False)

print(tensor.shape)
result = global_avg_pool_2d(tensor)
print(result.shape)
```
Output:
```
(10, 8, 5, 3)
(10, 8)

```

