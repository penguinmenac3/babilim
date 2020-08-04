[Back to Overview](../../../README.md)

# babilim.model.layers.convolution

> Convolution for 1d and 2d.

# *class* **Conv1D**(ModuleNative)

A 1d convolution layer.

* filters: The number of filters in the convolution. Defines the number of output channels.
* kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically 1, 3 or 5 are recommended.
* padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* stride: The offset between two convolutions that are applied. Typically 1. Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* dilation_rate: The dilation rate for a convolution.
* kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
* activation: The activation function that should be added after the dense layer.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

conv1d = Conv1D(filters=10, kernel_size=1)
tensor = Tensor(data=np.zeros((10,20,30), dtype=np.float32), trainable=False)

print(tensor.shape)
result = conv1d(tensor)
print(result.shape)
```
Output:
```
(10, 20, 30)
(10, 10, 30)

```

# *class* **Conv2D**(ModuleNative)

A 2d convolution layer.

* filters: The number of filters in the convolution. Defines the number of output channels.
* kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically (1,1) (3,3) or (5,5) are recommended.
* padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* stride: The offset between two convolutions that are applied. Typically (1, 1). Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* dilation_rate: The dilation rate for a convolution.
* kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
* activation: The activation function that should be added after the dense layer.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

conv2d = Conv2D(filters=10, kernel_size=(1, 1))
tensor = Tensor(data=np.zeros((10, 20, 5, 5), dtype=np.float32), trainable=False)

print(tensor.shape)
tensor = conv2d(tensor)
print(tensor.shape)
```
Output:
```
(10, 20, 5, 5)
(10, 10, 5, 5)

```

