# AUTOGENERATED FROM: babilim/data/dataset.ipynb

# Cell: 0
# MIT License
#
# Copyright (c) 2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Cell: 1
from typing import Any, Sequence, Tuple
import traceback
import os
import sys
import pickle
import babilim
from babilim.core.logging import info, status
from babilim.core.config import Config
from babilim.data.dataloader import Dataloader


# Cell: 2
class Dataset(Sequence):
    def __init__(self, config: Config, cache_dir: str = None):
        """
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
        
        :param config: The configuration used for your problem. (The problem parameters and train_batch_size are relevant for data loading.)
        :param cache_dir: The directory where the dataset can cache itself. Caching allows faster loading, when complex transformations are required.
        """
        self.config = config
        self.transformers = []
        self.realtime_transformers = []
        self._caching = False
        self._cache_dir = cache_dir
        self._cache_indices = {}
        self._cached_len = -1
        if self._cache_dir is not None:
            self.init_caching(cache_dir)
            self._cached_len = len(self._cache_indices)

    def init_caching(self, cache_dir):
        """
        Initialize caching for quicker access once the data was cached once.
        
        The caching caches the calls to the getitem including application of regular transformers.
        When calling this function the cache gets read if it exists or otherwise the folder is created and on first calling the getitem the item is stored.
        
        :param cache_dir: Directory where the cache should be stored.
        """
        info("Init caching: {}".format(cache_dir))
        self._caching = True
        self._cache_dir = cache_dir
        # If it does not exist create the cache dir.
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Read all files in the folder into a dict that maps indices to filenames (for quicker access)
        cache_files = os.listdir(self._cache_dir)
        for cf in cache_files:
            if cf.endswith(".pk"):
                self._cache_indices[int(cf.replace(".pk", ""))] = os.path.join(self._cache_dir, cf)

    def _cache(self, index: int, value) -> None:
        fp = os.path.join(self._cache_dir, "{:09d}.pk".format(index))
        with open(fp, "wb") as f:
            pickle.dump(value, f)
        self._cache_indices[index] = fp

    def getitem(self, index: int) -> Tuple[Any, Any]:
        """
        Gets called by `__getitem__`.
        
        This function must be overwritten by subclasses.
        It loads a training sample given an index in the dataset.
        
        Never overwrite `__getitem__` directly, as it handles the caching and application of transformers.
        
        :param index: The index between 0 and len(self), identifying the sample that should be loaded.
        :return: A tuple of features and values for the neural network. Features must be of type InputType (namedtuple) and labels of type InputType(namedtuple).
        """
        if self._caching:
            raise KeyError("The cache for index '{}' is missing.".format(index))
        else:
            raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Check if len is exceeded.
        if index >= len(self):
            raise IndexError()

        if self._caching and index in self._cache_indices:
            with open(self._cache_indices[index], "rb") as f:
                sample = pickle.load(f)
        else:
            # Print index errors, they probably were an error and not intentional.
            try:
                sample = self.getitem(index)
            except IndexError as e:
                traceback.print_exc(file=sys.stderr)
                raise e

            # Apply transforms if they are available.
            for transform in self.transformers:
                sample = transform(*sample)

            if self._caching:
                return self._cache(index, sample)

        # Apply real time transformers after caching. Realtime is not cached
        for transform in self.realtime_transformers:
            sample = transform(*sample)

        return sample

    def __len__(self) -> int:
        if self._cached_len >= 0:
            return self._cached_len
        raise NotImplementedError

    @property
    def version(self) -> str:
        """
        Property that returns the version of the dataset.
        
        **You must not overwrite this, instead overwrite `_get_version(self) -> str` used by this property.**

        :return: The version number of the dataset.
        """
        version = "{}".format(self._get_version())
        for transform in self.transformers:
            version = "{}_{}".format(version, transform.version)
        for transform in self.realtime_transformers:
            version = "{}_{}".format(version, transform.version)
        return version

    def _get_version(self) -> str:
        """
        Get the version string of the dataset.
        
        **Must be overwritten by every subclass.**

        :return: The version number of the dataset.
        """
        raise NotImplementedError

    def to_dataloader(self) -> Dataloader:
        """
        Converts the dataset into a babilim.data.Dataloader.
        
        :return: Returns a babilim dataloader object usable with the trainers.
        """
        data = None
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            data = self.to_pytorch()
        elif babilim.is_backend(babilim.TF_BACKEND):
            data = self.to_keras()
        else:
            raise NotImplementedError("Other backends than pytorch and tf2 are not implemented.")
        return Dataloader(data, self)

    def to_keras(self):
        """
        Converts the dataset into a batched keras dataset.
        
        You can use this if you want to use a babilim dataset without babilim natively in keras.
        
        :return: The type will be tf.keras.Sequence.
        """
        from babilim.data.keras import BatchedKerasDataset
        return BatchedKerasDataset(self, self.config)

    def to_pytorch(self):
        """
        Converts the dataset into a batched pytorch dataset.
        
        You can use this if you want to use a babilim dataset without babilim natively in pytorch.
        
        :return: The type will be torch.utils.data.DataLoader.
        """
        from babilim.data.pytorch import BatchedPytorchDataset
        return BatchedPytorchDataset(self, self.config, self.config.problem_shuffle, self.config.problem_num_threads)


    def to_tfrecord(self):
        """
        Creates a tfrecord dataset from the dataset.

        **Currently not implemented. Use Dataset.from_disk(...).to_keras() instead.**
        """
        raise NotImplementedError("This is not implemented yet. Use Dataset.from_disk(...).to_keras() instead.")

    def to_disk(self, cache_path: str, verbose: bool = True) -> None:
        """
        Write a dataset as a cache to the disk.
 
        :param cache_path: The path where the cache should be written.
        :param verbose: If info on progress should be printed, defaults to True.
        """
        self.init_caching(cache_path)
        if verbose:
            info("Caching dataset to {}".format(cache_path))
        N = len(self)
        for i, _ in enumerate(self):
            if verbose:
                status("{}/{}".format(i, N), end="")
        
        if verbose:
            info("")
            info("Caching done.")
 
    @staticmethod
    def from_disk(config: Config, cache_path: str) -> 'Dataset':
        """
        Create a dataset from a cache on disk.

        :param config: The configuration for the dataset.
        :param cache_path: The path to the cache.
        :param version: The version of the dataset that should be loaded.
        :return: A Dataset object that represents the data that has been passed to "to_disk" when creating the cache.
        """
        return Dataset(config, cache_dir=cache_path)
