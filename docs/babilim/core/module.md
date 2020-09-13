[Back to Overview](../../README.md)

# babilim.core.module

> An object which can have a state that is trainable or checkpointable. The core unit of babilim.

---
---
## *class* **Module**(object)

A module is an object with variables that can be trainable and checkpointable.

Furthermore every module is callable.
A module can be used with native or babilim tensors when the callable api is used.
It automatically wraps native tensors and calls the `call` function.

Attributes:
* `self.initialized_module`: A boolen storing if the module is already initialized. When not initialized loading state will fail.
* `self.device`: Specifies the device on which this module is.


---
### *def* **initialize**(*self*, dataset)

Initializes your module by running a sample of your dataset through it.

* **dataset**: The dataset you want to use for initialization. (Must be of type babilim.data.Dataset)




Makes a module callable. Automatically wraps tensorflow or pytorch tensors to ITensors from babilim.

```python
module = MyModule()
module(*args, **kwargs)
```

Warning: This function should not be overwritten. Instead overwrite `call` with no underscores.

* ***args**: All indexed parameters of your call function derivate.
* ****kwargs**: All named parameters of your call function derivate.


---
### *def* **build**(*self*, *args, **kwargs) -> None

This function will build your model and must be annotated with the RunOnlyOnce-Annotation.

Allocating weight tensors should be done here.
You can make use of the knowledge of your inputs to compute shapes for your weight tensors.
This will make coding dimensions a lot easier.

```python
@RunOnlyOnce
def build(self, image: ITensor) -> None:
```

* ***args**: You must specify the exact same parameters as for your call.
* ****kwargs**: You must specify the exact same parameters as for your call.


---
### *def* **call**(*self*, *args, **kwargs) -> Any

Makes a module callable and contains the forward pass of your model.
This should be pure computation and not allocate any weights.
Allocating weights should be done in the `build` function.

This function gets called by `__call__` and must be overwritten by any derived class.

```python
def call(self, image: ITensor) -> NetworkOutput:
```

* ***args**: You can specify any parameters you want.
* ****kwargs**: You can specify any named parameters you want.


---
### *def* **predict**(*self*, **kwargs)

Pass in single training examples as numpy arrays.
And predict the value without gradients.
Should be used for testing and evaluation.

If your network has eval modes you need to set them manually.

The array must not have batch dimension.

* **kwargs**: The parameters to feed the network as a single example.
* **returns**: The output for a single example.


---
### *def* **submodules**(*self*)

A property to get all submodules.

A submodule is a module stored in an attribute of a module.

```python
module.submodules
```


---
### *def* **modules**(*self*)

Returns an iterator over all submodules in the module.

A submodule is a module stored in an attribute of a module.


---
### *def* **named_modules**(*self*, memo=None, prefix='')

A named list of all submodules.

A submodule is a module stored in an attribute of a module.




Allows registration of the parameters with a native module.

This makes the parameters of a babilim modules available to the native modules.
When using a babilim modules in a native modules, use this function and pass the native module as a parameter.

This function works by adding all trainable_variables to the module you pass.
Warning: You need to build the babilim modules before calling this function. Building can be done by calling for example.

Here is a pytorch example:

```python
import torch
from torch.nn import Module
from babilim.modules import Linear

class MyModule(Module):
def __init__(self):
super().__init__()
self.linear = Linear(10)

def forward(self, features):
result = self.linear(features)
self.linear.register_params(self)
return result
```

* **module**: The native module on which parameters of this modules should be registered.


---
### *def* **training**(*self*) -> bool

Property if the object is in training mode.

```python
module.training
```

* **returns**: True if the object is in training mode.


---
### *def* **variables**(*self*)

Property with all variables of the object.

```python
module.variables
```

* **returns**: A list of the variables in this object.


---
### *def* **named_variables**(*self*)

Property with all variables of the object.

```python
module.named_variables
```

* **returns**: A dictionary of the variables in this object.


---
### *def* **trainable_variables**(*self*)

Property with trainable variables of the object.

```python
module.trainable_variables
```

* **returns**: A list of the trainable variables in this object.


---
### *def* **named_trainable_variables**(*self*)

Property with trainable variables of the object.

```python
module.named_trainable_variables
```

* **returns**: A dictionary of the trainable variables in this object.


---
### *def* **untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
module.untrainable_variables
```

* **returns**: A list of not trainable variables in this object.


---
### *def* **named_untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
module.named_untrainable_variables
```

* **returns**: A dictionary of not trainable variables in this object.


---
### *def* **trainable_variables_native**(*self*)

Property with not trainable variables of the object in native format.

```python
module.trainable_variables_native
```

* **returns**: A list of trainable variables in this object in native format.


---
### *def* **state_dict**(*self*) -> Dict

Get the state of the object as a state dict (usable for checkpoints).

* **returns**: A dictionary containing the state of the object.


---
### *def* **load_state_dict**(*self*, state_dict: Dict) -> None

Load the state of the object from a state dict.

Handy when loading checkpoints.

* **state_dict**: A dictionary containing the state of the object.


---
### *def* **eval**(*self*)

Set the object into eval mode.

```python
self.train(False)
```


---
### *def* **train**(*self*, mode=True)

Set the objects training mode.

* **mode**: (Optional) If the training mode is enabled or disabled. (default: True)


---
### *def* **load**(*self*, checkpoint_file_path: str) -> None

Load the state of the object from a checkpoint.

* **checkpoint_file_path**: The path to the checkpoint storing the state dict.


---
### *def* **save**(*self*, checkpoint_file_path: str) -> None

Save the state of the object to a checkpoint.

* **checkpoint_file_path**: The path to the checkpoint storing the state dict.


