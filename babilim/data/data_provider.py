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
from typing import Iterable, List, Any, Sequence, Iterator
import traceback
import os
import sys
import pickle

import babilim
from babilim import info
from babilim.core.config import Config
from babilim.core.tensor import TensorWrapper


class Dataloader(Iterable):
    def __init__(self, native_dataloader):
        self._tensor_wrapper = TensorWrapper()
        self.native_dataloader = native_dataloader

    def __iter__(self) -> Iterator:
        class TensorDataloaderIterator(Iterator):
            def __init__(self, native_dataloader, tensor_wrapper):
                self._tensor_wrapper = tensor_wrapper
                self.native_dataloader_iter = iter(native_dataloader)

            def __next__(self) -> Any:
                # Print index errors, they probably were an error and not intentional.
                try:
                    x, y = next(self.native_dataloader_iter)
                    inp, _ = self._tensor_wrapper.wrap(x._asdict())
                    outp, _ = self._tensor_wrapper.wrap(y._asdict())
                    inp = type(x)(**inp)
                    outp = type(y)(**outp)
                    return inp, outp
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return TensorDataloaderIterator(self.native_dataloader, self._tensor_wrapper)

    def __len__(self) -> int:
        return len(self.native_dataloader)


class Dataset(Sequence):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    Extending on the pytorch dataset this dataset also needs to implement a ``version`` function.
    The version function returns a number (can be a hash) which changes, whenever the dataset changes.
    This enables subsequent callers to buffer this dataset and update their buffers when the version changes.
    """
    def __init__(self, config: Config, cache_dir: str = None):
        self.config = config
        self.transformers = []
        self.realtime_transformers = []
        self._caching = False
        self._cache_dir = cache_dir
        self._cache_indices = {}
        self._cached_len = -1
        if self._cache_dir is not None:
            self.init_caching(cache_dir)
            self._cached_len = len(os.listdir(self._cache_dir))

    def init_caching(self, cache_dir):
        info("Init caching: {}".format(cache_dir))
        self._caching = True
        self._cache_dir = cache_dir
        # If it does not exist create the cache dir.
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Read all files in the folder into a dict that maps indices to filenames (for quicker access)
        cache_files = os.listdir(self._cache_dir)
        for cf in cache_files:
            self._cache_indices[int(cf.replace(".pk", ""))] = os.path.join(self._cache_dir, cf)

    def _cache(self, index: int, value) -> None:
        fp = os.path.join(self._cache_dir, "{:09d}.pk".format(index))
        with open(fp, "wb") as f:
            pickle.dump(value, f)
        self._cache_indices[index] = fp

    def getitem(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
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
        Defines the version of the data in the dataset. When the data is static you can return a static string.

        :return: The version number of the dataset.
        """
        version = "{}".format(self._get_version())
        for transform in self.transformers:
            version = "{}_{}".format(version, transform.version)
        for transform in self.realtime_transformers:
            version = "{}_{}".format(version, transform.version)
        return version

    def _get_version(self):
        """
        Defines the version of the data in the dataset. When the data is static you can return a static string.

        :return: The version number of the dataset.
        """
        raise NotImplementedError

    def to_disk(self, cache_path: str, verbose: bool = True) -> None:
        """
        Write a dataset as a cache to the disk.
 
        :param cache_path: The path where the cache should be written.
        :param verbose: If info on progress should be printed, defaults to True.
        """
        self.init_caching(cache_path)
        if verbose:
            info("Caching dataset...")
        N = len(self)
        for i, _ in enumerate(self):
            if verbose:
                info("\r{}/{}".format(i, N), end="")
        
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
        :return: A CachedDataset object that represents the data that has been passed to "to_disk" when creating the cache.
        """
        return Dataset(config, cache_dir=cache_path)

    def to_keras(self):
        """
        Converts the dataset into a batched keras dataset.
        
        The type will be tf.keras.Sequence.
        """
        from babilim.data.keras import BatchedKerasDataset
        return BatchedKerasDataset(self, self.config)

    def to_pytorch(self):
        """
        Converts the dataset into a batched pytorch dataset.
        
        The type will be torch.utils.data.DataLoader.
        """
        from babilim.data.pytorch import BatchedPytorchDataset
        return BatchedPytorchDataset(self, self.config, self.config.problem_shuffle, self.config.problem_num_threads)

    def to_dataloader(self) -> Dataloader:
        """
        Converts the dataset into a babilim.data.Dataloader.
        """
        data = None
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            data = self.to_pytorch()
        elif babilim.is_backend(babilim.TF_BACKEND):
            data = self.to_keras()
        else:
            raise NotImplementedError("Other backends than pytorch and tf2 are not implemented.")
        return Dataloader(data)

    def to_tfrecord(self):
        """
        Creates a tfrecord dataset from the dataset.

        .. warning::

            Currently not implemented. Use .as_cached(...).to_keras() instead.

        """
        raise NotImplementedError("This is not implemented yet. Use dataset.as_cached(...).to_keras() instead.")


class Transformer(object):
    """
    A transformer should implement ``__call__``.
    """
    def __call__(self, *args):
        """
        This function gets the data from the previous transformer or dataset as input and should output the data again.
        :param args: The input data.
        :return: The output data.
        """
        raise NotImplementedError

    @property
    def version(self):
        """
        Defines the version of the transformer. The name can be also something descriptive of the method.

        :return: The version number of the transformer.
        """
        raise NotImplementedError
