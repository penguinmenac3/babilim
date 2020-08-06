[Back to Overview](../../README.md)

# babilim.model.sequential

> Sequentially combine modules into a model.

---
---
## *class* **Sequential**(Module)

Create a module which is a sequential order of other layers.

Runs the layers in order.

```python
my_seq = Sequential(layer1, layer2, layer3)
```

* **layers**: All ordered parameters are used as layers.


---
### *def* **call**(*self*, features)

Do not call this directly, use `__call__`:
```
my_seq(features)
```


Example:
```python
from babilim.core.tensor import Tensor
from babilim.model.layers.convolution import Conv2D
import numpy as np

conv1 = Conv2D(filters=10, kernel_size=(1,1))
conv2 = Conv2D(filters=3, kernel_size=(1,1))

my_seq = Sequential(conv1, conv2)

tensor = Tensor(data=np.zeros((10,8,8,3), dtype=np.float32), trainable=False)
print(tensor.shape)
result = my_seq(tensor)
print(result.shape)
```
Output:
```
(10, 8, 8, 3)
(10, 3, 8, 3)

```

