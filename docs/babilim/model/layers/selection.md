[Back to Overview](../../../README.md)

# babilim.model.layers.selection

> These layers select parts of a tensor.

---
---
## *class* **Gather**(ModuleNative)

Gather tensors from one tensor by providing an index tensor.

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[N, L, ?]) The tensor from which to gather values at the given indices.
* **indices**: (Tensor[N, K]) The indices at which to return the values of the input tensor.
* **returns**: (Tensor[N, K, ?]) The tensor containing the values at the indices given.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

gather = Gather()
tensor = Tensor(data=np.zeros((2,8,3), dtype=np.float32), trainable=False)
indices = Tensor(data=np.array([[6,3], [1,2]]), trainable=False)

print(tensor.shape)
result = gather(tensor, indices.cast("int64"))
print(result.shape)
```
Output:
```
(2, 8, 3)
(2, 2, 3)

```

---
---
## *class* **TopKIndices**(ModuleNative)

Returns the top k tensor indices (separate per batch).

Created object is callable with the following parameters:
* **input_tensor**: (Tensor[N, L]) The tensor in which to search the top k indices.
* **returns**: (Tensor[N, K]) The tensor containing the indices of the top k values.

Parameters for the constructor:
* **k**: The number of indices to return per batch.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

top3 = TopKIndices(k=3)
tensor = Tensor(data=np.zeros((2,8), dtype=np.float32), trainable=False)

print(tensor.shape)
result = top3(tensor)
print(result.shape)
```
Output:
```
(2, 8)
(2, 3)

```

