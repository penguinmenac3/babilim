# babilim.core.config

> The base class for every config.

This code is under the MIT License.

# *class* **Config**(object)

A configuration for a deep learning project.

This class should never be instantiated directly, subclass it instead and add your atributes after calling super.


Every configuration has these filds, which you may overwrite as you need.

### Dataset/Problem Parameters
* `self.problem_base_dir = None`: The path to the root of the dataset folder.
* `self.problem_shuffle = True`: If the dataloader used for training should shuffle the data.
* `self.problem_num_threads = 0`: How many threads the dataloader should use. (0 means no multithreading and is most stable)

### Training Parameters
* `self.train_batch_size = 1`: The batch size used for training the neural network. This is required for the dataloader from the dataset.
* `self.train_epochs = 1`: The number epochs for how many a training should run.

Example:
```python
class MyConfig(Config):
    def __init__(self, problem_base_dir: str) -> None:
        """
        This is my example configuration for X.
        
        :param problem_base_dir: The path to the root of the dataset folder.
        """
        super().__init__()
        
        self.problem_base_dir = problem_base_dir
        self.problem_dataset = MyDatasetClass
        self.problem_my_param = 42
        
        self.train_batch_size = 32
        self.train_epochs = 50
        self.train_my_param = 1337
```

# Dynamic Config Import

When you write a library and need to dynamically import configs, use the following two functions.

### *def* **import_config**(config_file: str) -> Config

Only libraries should use this method. Human users should directly import their configs.
Automatically imports the most specific config from a given file.

* config_file: Path to the configuration file (e.g. configs/my_config.py)
* returns: The configuration object.


### *def* **import_checkpoint_config**(config_file: str) -> Any

Adds the folder in which the config_file is to the pythonpath, imports it and removes the folder from the python path again.

* config_file: The configuration file which should be loaded.
* returns: The configuration object.


