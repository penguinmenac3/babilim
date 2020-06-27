# babilim.core.statefull_object

> An object which can have a state that is trainable or checkpointable. The core unit of babilim.

# *class* **StatefullObject**(object)

A statefull object is an object with variables that can be trainable and checkpointable.


### *def* **training**(*self*) -> bool

Property if the object is in training mode.

```python
statefull_object.training
```

* returns: True if the object is in training mode.


### *def* **variables**(*self*)

Property with all variables of the object.

```python
statefull_object.variables
```

* returns: A list of the variables in this object.


### *def* **named_variables**(*self*)

Property with all variables of the object.

```python
statefull_object.named_variables
```

* returns: A dictionary of the variables in this object.


### *def* **trainable_variables**(*self*)

Property with trainable variables of the object.

```python
statefull_object.trainable_variables
```

* returns: A list of the trainable variables in this object.


### *def* **named_trainable_variables**(*self*)

Property with trainable variables of the object.

```python
statefull_object.named_trainable_variables
```

* returns: A dictionary of the trainable variables in this object.


### *def* **untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
statefull_object.untrainable_variables
```

* returns: A list of not trainable variables in this object.


### *def* **named_untrainable_variables**(*self*)

Property with not trainable variables of the object.

```python
statefull_object.named_untrainable_variables
```

* returns: A dictionary of not trainable variables in this object.


### *def* **trainable_variables_native**(*self*)

Property with not trainable variables of the object in native format.

```python
statefull_object.trainable_variables_native
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


