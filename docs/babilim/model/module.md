# babilim.model.module

> The base class for every module.

This code is under the MIT License.

# *class* **Module**(StatefullObject)

A module is the base building block of all models and layers.

Users must overwrite `__init__`,`call` and `build`. For the `build` do not forget the `@RunOnlyOnce`-Annotation.
The rest comes pre implemented and should not be overwritten.
When overwriting the `__init__` call the super init: `super().__init__(layer_type)`.

* layer_type: The type of the layer provided.


### *def* **initialize**(*self*, dataset: Dataset)

Initializes your module by running a sample of your dataset through it.

* dataset: The dataset you want to use for initialization. (Must be of type babilim.data.Dataset)


### *def* **predict**(*self*, **kwargs)

Pass in single training examples as numpy arrays.
And predict the value without gradients.
Should be used for testing and evaluation.

If your network has eval modes you need to set them manually.

The array must not have batch dimension.

* kwargs: The parameters to feed the network as a single example.
* returns: The output for a single example.


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


### *def* **layer_type**(*self*)

A property to get the layer type.

```python
module.layer_type
```


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


