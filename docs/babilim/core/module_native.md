# babilim.core.module_native

> A module that is implemented by native function calls.

# *class* **ModuleNative**(Module)

A module with a native implementation.

This module is like a normal module, except that call and build call a "call_pytorch", "call_tf", "build_pytorch" and "build_tf" depending on what backend is set.


### *def* **build**(*self*, *args, **kwargs) -> None

Build the model, this function automatically calls the native build with the tensors unwrapped.

* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


### *def* **build_pytorch**(*self*, *args, **kwargs) -> None

A native build function in pytorch.

Even though babilim never calls this function directly multiple times, it is recommended to add the RunOnlyOnce guard in case a user calls it multiple times.

* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


### *def* **build_tf**(*self*, *args, **kwargs) -> None

A native build function in tensorflow.

Even though babilim never calls this function directly multiple times, it is recommended to add the RunOnlyOnce guard in case a user calls it multiple times.

* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


### *def* **call**(*self*, *args, **kwargs) -> Any

Makes a module callable and contains the forward pass of your model.
This should be pure computation and not allocate any weights.
Allocating weights should be done in the `build` function.

This function gets called by `__call__` and must be overwritten by any derived class.

```python
def call(self, image: ITensor) -> NetworkOutput:
```

* *args: You can specify any parameters you want.
* **kwargs: You can specify any named parameters you want.


### *def* **call_pytorch**(*self*, *args, **kwargs) -> Any

A native call function in pytorch (like the forward).

* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


### *def* **build_tf**(*self*, *args, **kwargs) -> Any

A native call function in tensorflow.

* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


