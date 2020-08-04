[Back to Overview](../../README.md)

# babilim.core.device

> Controll on what device code is executed.

### *def* **get_current_device**() -> str

Get a string specifying the currently selected default device.

When you manually assign a device, you should always use this device.


### *def* **get_current_device_native_format**() -> str

Get a string specifying the currently selected default device in the backend specific native format.

When you manually assign a device, you should always use this device.


# *class* **Device**(object)

Set the default device for babilim in a with statement.

```python
with Device("gpu:1"):
# All tensors of MyModule are on gpu 1 automatically.
mymodule = MyModule()
```

When there is nested with-device statements, the innermost overwrites all others.
By default gpu:0 is used.

Only works for tensors which are at some point wrapped by a babilim module (Lambda, Tensor, etc.).

* name: The name of the device. ("cpu", "gpu:0", "gpu:1", etc.)


