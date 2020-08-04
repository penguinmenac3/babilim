[Back to Overview](../../README.md)

# babilim.data.dataset

> A base class for implementing datasets with ease.

# *class* **Dataset**(Sequence)

An abstract class representing a Dataset.

All other datasets must subclass it. All subclasses must override
`__len__`, that provides the size of the dataset, and `getitem`,
supporting integer indexing in range from 0 to len(self) exclusive and `_get_version`.

Extending on the pytorch dataset this dataset also needs to implement a `version` function.
The version function returns a number (can be a hash) which changes, whenever the dataset changes.
This enables subsequent callers to buffer this dataset and update their buffers when the version changes.

A dataset loads the data from the disk as general as possible and then transformers adapt it to the needs of the neural network.
There are two types of transformers (which are called in the order listed here):
* `self.transformers = []`: These transformers are applied once on the dataset (before caching is done).
* `self.realtime_transformers = []`: These transformers are applied every time a sample is retrieved. (e.g. random data augmentations)

* config: The configuration used for your problem. (The problem parameters and train_batch_size are relevant for data loading.)
* cache_dir: The directory where the dataset can cache itself. Caching allows faster loading, when complex transformations are required.


### *def* **init_caching**(*self*, cache_dir)

Initialize caching for quicker access once the data was cached once.

The caching caches the calls to the getitem including application of regular transformers.
When calling this function the cache gets read if it exists or otherwise the folder is created and on first calling the getitem the item is stored.

* cache_dir: Directory where the cache should be stored.


### *def* **getitem**(*self*, index: int) -> Tuple[Any, Any]

Gets called by `__getitem__`.

This function must be overwritten by subclasses.
It loads a training sample given an index in the dataset.

Never overwrite `__getitem__` directly, as it handles the caching and application of transformers.

* index: The index between 0 and len(self), identifying the sample that should be loaded.
* returns: A tuple of features and values for the neural network. Features must be of type InputType (namedtuple) and labels of type InputType(namedtuple).


### *def* **version**(*self*) -> str

Property that returns the version of the dataset.

**You must not overwrite this, instead overwrite `_get_version(self) -> str` used by this property.**

* returns: The version number of the dataset.


### *def* **to_dataloader**(*self*) -> Dataloader

Converts the dataset into a babilim.data.Dataloader.

* returns: Returns a babilim dataloader object usable with the trainers.


### *def* **to_keras**(*self*)

Converts the dataset into a batched keras dataset.

You can use this if you want to use a babilim dataset without babilim natively in keras.

* returns: The type will be tf.keras.Sequence.


### *def* **to_pytorch**(*self*)

Converts the dataset into a batched pytorch dataset.

You can use this if you want to use a babilim dataset without babilim natively in pytorch.

* returns: The type will be torch.utils.data.DataLoader.


### *def* **to_tfrecord**(*self*)

Creates a tfrecord dataset from the dataset.

**Currently not implemented. Use Dataset.from_disk(...).to_keras() instead.**


### *def* **to_disk**(*self*, cache_path: str, verbose: bool = True) -> None

Write a dataset as a cache to the disk.

* cache_path: The path where the cache should be written.
* verbose: If info on progress should be printed, defaults to True.


### *def* **from_disk**(config: Config, cache_path: str) -> 'Dataset'

Create a dataset from a cache on disk.

* config: The configuration for the dataset.
* cache_path: The path to the cache.
* version: The version of the dataset that should be loaded.
* returns: A Dataset object that represents the data that has been passed to "to_disk" when creating the cache.


