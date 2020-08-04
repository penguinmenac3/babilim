[Back to Overview](../../../README.md)

# babilim.model.layers.batch_normalization

> Apply batch normalization to a tensor.

# *class* **BatchNormalization**(ModuleNative)

A batch normalization layer.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

batch_norm = BatchNormalization()
tensor = Tensor(data=np.array([[10,3,-4,2], [5, 5, 4, -2], [1,-7,2,0]], dtype=np.float32), trainable=False)

print(tensor.shape)
print(tensor)
result = batch_norm(tensor)
print(tensor.shape)
print(result)
```
Output:
```
(3, 4)
tensor([[10.,  3., -4.,  2.],
        [ 5.,  5.,  4., -2.],
        [ 1., -7.,  2.,  0.]], device='cuda:0')
(3, 4)
tensor([[ 1.2675,  0.5080, -1.3728,  1.2247],
        [-0.0905,  0.8890,  0.9806, -1.2247],
        [-1.1770, -1.3970,  0.3922,  0.0000]], device='cuda:0',
       grad_fn=<CudnnBatchNormBackward>)

```

