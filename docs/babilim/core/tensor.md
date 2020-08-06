[Back to Overview](../../README.md)

# babilim.core.tensor

> Create a tensor independent of the underlying framework.

# Create Tensors

This package creates tensors of type ITensor.
It does not contain any classes but just a function which creates tensors.
This is because there is different implementations of the ITensor interface for pytorch and tensorflow required.

---
### *def* **Tensor**(*, data: Union[np.ndarray, Any], trainable: bool) -> I**Tensor**

Create a babilim tensor from a native tensor or numpy array.

* **data**: The data that should be put in a babilim tensor. This can be either a numpy array or a pytorch/tensorflow tensor.
* **trainable**: If the tensor created should be trainable. Only works for numpy tensors, native tensors overwrite this field!
* **returns**: An object of type babilim.core.ITensor.


---
### *def* **TensorWrapper**() -> I**TensorWrapper**

Create a tensor wrapper object.

Sometimes it is nescesarry to implement stuff in native pytorch or native tensorflow. Here the tensor wrapper can help.

**WARNING: Instead of directly using the TensorWrapper, you should prefer using the babilim.module.Lambda!**


