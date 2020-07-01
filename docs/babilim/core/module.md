# babilim.core.module

> An object which can have a state that is trainable or checkpointable. The core unit of babilim.

# *class* **Module**(object)

A module is an object with variables that can be trainable and checkpointable.

Furthermore every module is callable.
A module can be used with native or babilim tensors when the callable api is used.
It automatically wraps native tensors and calls the `call` function.

Attributes:
* `self.initialized_module`: A boolen storing if the module is already initialized. When not initialized loading state will fail.
* `self.device`: Specifies the device on which this module is.


### *def* **initialize**(*self*, dataset: Dataset)

Initializes your module by running a sample of your dataset through it.

* dataset: The dataset you want to use for initialization. (Must be of type babilim.data.Dataset)


### *def* **build**(*self*, *args, **kwargs) -> None

This function will build your model and must be annotated with the RunOnlyOnce-Annotation.

Allocating weight tensors should be done here.
You can make use of the knowledge of your inputs to compute shapes for your weight tensors.
This will make coding dimensions a lot easier.

```python
@RunOnlyOnce
def build(self, image: ITensor) -> None:
```

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


### *def* **predict**(*self*, **kwargs)

Pass in single training examples as numpy arrays.
And predict the value without gradients.
Should be used for testing and evaluation.

If your network has eval modes you need to set them manually.

The array must not have batch dimension.

* kwargs: The parameters to feed the network as a single example.
* returns: The output for a single example.


### *def* **submodules**(*self*)

A property to get all submodules.

A submodule is a module stored in an attribute of a module.

```python
module.submodules
```


### *def* **modules**(*self*)

Returns an iterator over all submodules in the module.

A submodule is a module stored in an attribute of a module.


### *def* **named_modules**(*self*, memo=None, prefix='')

A named list of all submodules.

A submodule is a module stored in an attribute of a module.


### *class* **MyModule**(Module)

*(no documentation found)*

### *def* **forward**(*self*, features)

if babilim.is_backend(PYTORCH_BACKEND):
from torch.nn import Module
if isinstance(module, Module):
myname = "_error_"
for var in module.__dict__:
if module.__dict__[var] == self:
myname = var
if isinstance(module.__dict__[var], list) and self in module.__dict__[var]:
myname = "{}/{}".format(var, module.__dict__[var].index(self))

# Register self as pytorch module.
module._modules[myname] = self

for name, param in self.named_variables.items():
if param.trainable:
module.register_parameter(myname + name, param.native)
else:
module.register_buffer(myname + name, param.native)
else:
if babilim.core.logging.DEBUG_VERBOSITY:
_warn_once("babilim.model.module.Module:_register_params Not implemented for tf2 but I think it is not required.")

@property
def training(self) -> bool:


### *def* **variables**(*self*)

Property with all variables of the object.

```python
module.variables
```

* returns: A list of the variables in this object.


### *def* **named_variables**(*self*)

Property with all variables of the object.

```python
module.named_variables
```

* returns: A dictionary of the variables in this object.


### *def* **trainable_variables**(*self*)

Property with trainable variables of the object.

```python
module.trainable_variables
```

* returns: A list of the trainable variables in this object.


### *def* **named_trainable_variables**(*self*)

Property with trainable variables of the object.

```python
module.named_trainable_variables
```

* returns: A dictionary of the trainable variables in this object.


### *def* **untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
module.untrainable_variables
```

* returns: A list of not trainable variables in this object.


### *def* **named_untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
module.named_untrainable_variables
```

* returns: A dictionary of not trainable variables in this object.


### *def* **trainable_variables_native**(*self*)

Property with not trainable variables of the object in native format.

```python
module.trainable_variables_native
```

* returns: A list of trainable variables in this object in native format.


### *def* **state_dict**(*self*) -> Dict

Get the state of the object as a state dict (usable for checkpoints).

* returns: A dictionary containing the state of the object.


### *def* **load_state_dict**(*self*, state_dict: Dict) -> None

Load the state of the object from a state dict.

Handy when loading checkpoints.

* state_dict: A dictionary containing the state of the object.


### *def* **eval**(*self*)

Set the object into eval mode.

```python
self.train(False)
```


### *def* **train**(*self*, mode=True)

Set the objects training mode.

* mode: (Optional) If the training mode is enabled or disabled. (default: True)


### *def* **load**(*self*, checkpoint_file_path: str) -> None

Load the state of the object from a checkpoint.

* checkpoint_file_path: The path to the checkpoint storing the state dict.


### *def* **save**(*self*, checkpoint_file_path: str) -> None

Save the state of the object to a checkpoint.

* checkpoint_file_path: The path to the checkpoint storing the state dict.


