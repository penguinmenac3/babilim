[Back to Overview](../../README.md)

# babilim.core.module_native

> A module that is implemented by native function calls.

# *class* **ModuleNative**(Module)

A module with a native implementation.

This module is like a normal module, except that call and build call a "call_pytorch", "call_tf", "build_pytorch" and "build_tf" depending on what backend is set.


### *def* **build**(*self*, *args, **kwargs) -> None

Build the model, this function automatically calls the native build with the tensors unwrapped.

This function gets called by `__call__` and itself passes all calls to `_build_pytorch` and `_build_tf`.
Furthermore, it takes care of unwrapping the tensors into native tensors before calling and wrapping them again after calling.
This allows the native functions `_build_pytorch` and `_build_tf` to be pure pytorch or tensorflow code.
All subclasses must implement `_build_pytorch` and `_build_tf`.

You should never call the build function directly. Call this module in the following style (this ensures the module is build on first run):
```
module = MyModule()
result = module(*args, **kwargs)  # <- Build gets called internally here.
```

Parameters:
* *args: You must specify the exact same parameters as for your call.
* **kwargs: You must specify the exact same parameters as for your call.


### *def* **call**(*self*, *args, **kwargs) -> Any

Makes a module callable and contains the forward pass of your model.
This should be pure computation and not allocate any weights.
Allocating weights should be done in the `build` function.

This function gets called by `__call__` and itself passes all calls to `_call_pytorch` and `_call_tf`.
Furthermore, it takes care of unwrapping the tensors into native tensors before calling and wrapping them again after calling.
This allows the native functions `_call_pytorch` and `_call_tf` to be pure pytorch or tensorflow code.
All subclasses must implement `_call_pytorch` and `_call_tf`.

You should call this module in the following style (this ensures the module is build on first run):
```
module = MyModule()
result = module(*args, **kwargs)
```

Parameters:
* *args: You can specify any parameters you want.
* **kwargs: You can specify any named parameters you want.


