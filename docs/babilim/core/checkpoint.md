[Back to Overview](../../README.md)

# babilim.core.checkpoint

> Loading and saving checkpoints with babilim.

### *def* **load_state**(checkpoint_path: str, native_format: bool = False) -> Dict

Load the state from a checkpoint.

* checkpoint_path: The path to the file in which the checkpoint is stored.
* native_format: (Optional) If the checkpoint should use the backend specific native format. (default: False)
* returns: A dict containing the states.


### *def* **save_state**(data, checkpoint_path, native_format=False)

Save the state to a checkpoint.

* data: A dict containing the states.
* checkpoint_path: The path to the file in which the checkpoint shall be stored.
* native_format: (Optional) If the checkpoint should use the backend specific native format. (default: False)


