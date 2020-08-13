[Back to Overview](../../../README.md)

# babilim.model.layers.native_wrapper

> Wrap a layer, function or model from native into babilim.

---
---
## *class* **Lambda**(Module)

Wrap a natively implemented layer into a babilim layer.

This can be used to implement layers that are missing in babilim in an easy way.

```
my_lambda = Lambda(tf.max)
```

* **native_module**: The native pytorch/tensorflow module that should be wrapped.
* **to_gpu**: (Optional) True if the module should be automatically be moved to the gpu. (default: True)


---
### *def* **build**(*self*, *args, **kwargs) -> None

*(no documentation found)*

---
### *def* **call**(*self*, *args, **kwargs) -> Any

Do not call this directly, use `__call__`:
```
my_lambda(*args, **kwargs)
```


---
### *def* **eval**(*self*)

*(no documentation found)*

---
### *def* **train**(*self*, mode=True)

*(no documentation found)*

Example:
```python
from babilim.core.tensor import Tensor
import numpy as np
from torch.nn import Conv1d

native_conv = Conv1d(in_channels=8, out_channels=3, kernel_size=(1,1))
my_lambda = Lambda(native_conv)
tensor = Tensor(data=np.zeros((10,8,8,3), dtype=np.float32), trainable=False)

print(tensor.shape)
result = my_lambda(tensor)
print(result.shape)
print(my_lambda.named_trainable_variables.keys())
```
Output:
```
(10, 8, 8, 3)
(10, 3, 8, 3)
dict_keys(['/native_module/weight', '/native_module/bias'])

```

