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
from babilim.experiment.config import Config
from babilim.core.tensor import TensorWrapper


class TensorDataset(Iterable):
    def __init__(self, native_dataset):
        self._tensor_wrapper = TensorWrapper()
        self.native_dataset = native_dataset
        self.native_dataset_iter = iter(native_dataset)

    def __iter__(self) -> Iterator:
        class TensorDatasetIterator(Iterator):
            def __init__(self, native_dataset, tensor_wrapper):
                self._tensor_wrapper = tensor_wrapper
                self.native_dataset_iter = iter(native_dataset)

            def __next__(self) -> Any:
                # Print index errors, they probably were an error and not intentional.
                try:
                    x, y = next(self.native_dataset_iter)
                    inp, _ = self._tensor_wrapper.wrap(x._asdict())
                    outp, _ = self._tensor_wrapper.wrap(y._asdict())
                    inp = type(x)(**inp)
                    outp = type(y)(**outp)
                    return inp, outp
                except IndexError as e:
                    traceback.print_exc(file=sys.stderr)
                    raise e
        return TensorDatasetIterator(self.native_dataset, self._tensor_wrapper)

    def __len__(self) -> int:
        return len(self.native_dataset)


class Dataset(Sequence):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.

    Extending on the pytorch dataset this dataset also needs to implement a ``version`` function.
    The version function returns a number (can be a hash) which changes, whenever the dataset changes.
    This enables subsequent callers to buffer this dataset and update their buffers when the version changes.
    """
    def __init__(self, config: Config):
        self.config = config
    
    def getitem(self, index: int) -> Any:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        # Check if len is exceeded.
        if index >= len(self):
            raise IndexError()

        # Print index errors, they probably were an error and not intentional.
        try:
            return self.getitem(index)
        except IndexError as e:
            traceback.print_exc(file=sys.stderr)
            raise e

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def version(self) -> str:
        """
        Defines the version of the data in the dataset. When the data is static you can return a static string.

        :return: The version number of the dataset.
        """
        raise NotImplementedError

    def as_cached(self, cache_path: str) -> '_CachedDataset':
        """
        Convert a dataset into a cached dataset.

        :param cache_path: The path where the cache should be saved.
        """
        if isinstance(self, _CachedDataset):
            raise RuntimeError("Cannot cache a cached dataset.")
        return _CachedDataset(dataset=self, cache_path=cache_path, version=self.version)

    def to_disk(self, cache_path: str, verbose: bool = True) -> None:
        """
        Write a dataset as a cache to the disk.
 
        :param cache_path: The path where the cache should be written.
        :param verbose: If info on progress should be printed, defaults to True.
        """
        dataset = self.as_cached(cache_path)
        if verbose:
            print("Caching dataset...")
        N = len(dataset)
        for i, _ in enumerate(dataset):
            if verbose:
                print("\r{}/{}".format(i, N), end="")
        
        if verbose:
            print()
            print("Caching done.")
 
    @staticmethod
    def from_disk(config: Config, cache_path: str, version: str) -> '_CachedDataset':
        """
        Create a dataset from a cache on disk.
 
        :param config: The configuration for the dataset.
        :param cache_path: The path to the cache.
        :param version: The version of the dataset that should be loaded.
        :return: A CachedDataset object that represents the data that has been passed to "to_disk" when creating the cache.
        """
        return _CachedDataset(dataset=_CacheDummyDataset(config=config, cache_path=cache_path, version=version), cache_path=cache_path, version=version)

    def to_keras(self):
        """
        Converts the dataset into a batched keras dataset.
        
        The type will be tf.keras.Sequence.
        """
        from babilim.data.keras import BatchedKerasDataset
        return BatchedKerasDataset(self, self.config)

    def to_pytorch(self, shuffle: bool = True, num_workers: int = 1):
        """
        Converts the dataset into a batched pytorch dataset.
        
        The type will be torch.utils.data.DataLoader.
        
        :param shuffle: If the data should be shuffeled. Defaults to True.
        :param num_workers: The number of multithreaded workers. Defaults to 1, since more usually does not work.
        """
        from babilim.data.pytorch import BatchedPytorchDataset
        return BatchedPytorchDataset(self, self.config, shuffle, num_workers)

    def to_native(self) -> TensorDataset:
        """
        TODO
        """
        data = None
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            data = self.to_pytorch()
        elif babilim.is_backend(babilim.TF_BACKEND):
            data = self.to_keras()
        else:
            raise NotImplementedError("Other backends than pytorch and tf2 are not implemented.")
        return TensorDataset(data)

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


class ComposeTransforms(Transformer):
    def __init__(self, transforms: Iterable[Transformer]) -> None:
        """
        A transform that applies the transforms provided in transforms in order.

        :param transforms: An Iterable of Transformers which is applied on the data.
        """
        self.transforms = transforms

    def __call__(self, *args):
        """
        Applies all the transforms in order.
        :param args: The input data.
        :return: The transformed data.
        """
        for t in self.transforms:
            args = t(*args)
        return args


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transformer: Transformer) -> None:
        """
        Create a transfored dataset by applying a transformer.

        :param dataset: The dataset to transform.
        :param transformer: The transformer that gets applied to the dataset.
        """
        super().__init__(dataset.config)
        self.dataset = dataset
        self.transformer = transformer
    
    def __len__(self) -> int:
        return len(self.dataset)

    def getitem(self, index: int) -> Any:
        return self.transformer(*self.dataset[index])

    @property
    def version(self) -> str:
        """
        Defines the version of the data in the dataset. When the data is static you can return a static string.

        :return: The version number of the dataset.
        """
        return "TransformedDataset_{}".format(self.dataset.version)

  
class _CacheDummyDataset(Dataset):
    def __init__(self, config: Config, cache_path: str, version: str) -> None:
        """
        A dummy dataset that can be used when a dataset has been fully cached.
 
        :param config: The configuration for the dataset.
        :param version: Version of the cached dataset that should be used.
        :type version: str
        """
        super().__init__(config)
        self._version = version
        self.cache_path = os.path.join(cache_path, version)
        self.size = len(os.listdir(self.cache_path))
 
    def getitem(self, index: int) -> Any:
        raise RuntimeError("This function should never be called on the dummy dataset.")
 
    def __len__(self) -> int:
        return self.size
 
    @property
    def version(self) -> str:
        """
        Defines the version of the data in the dataset. When the data is static you can return a static string.
        :return: The version number of the dataset.
        """
        return self._version
 
 
class _CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, cache_path: str, version: str):
        """
        It is not recommended to use this class. Use the static methods to_disk and from_disk instead.
 
        A cached dataset stores the data provided by the original dataset on disk in a quick to read manner.
        When you do computation heavy preprocessing using a cached dataset might be a great idea.
 
        :param dataset: The dataset that should be wrapped.
        """
        super().__init__(dataset.config)
        self.dataset = dataset
        self._version = version
        self.cache_dir = os.path.join(cache_path, version)
        self.cache_indices = {}
 
        # If it does not exist create the cache dir.
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
 
        # Read all files in the folder into a dict that maps indices to filenames (for quicker access)
        cache_files = os.listdir(self.cache_dir)
        for cf in cache_files:
            self.cache_indices[int(cf.replace(".pk", ""))] = os.path.join(self.cache_dir, cf)
 
    def _cache(self, index: int) -> None:
        fp = os.path.join(self.cache_dir, "{:09d}.pk".format(index))
        value = self.dataset[index]
        with open(fp, "wb") as f:
            pickle.dump(value, f)
        self.cache_indices[index] = fp
        return value
 
    def getitem(self, index: int) -> Any:
        if index in self.cache_indices:
            with open(self.cache_indices[index], "rb") as f:
                return pickle.load(f)
        else:
            return self._cache(index)
 
    def __len__(self) -> int:
        return len(self.dataset)
 
    @property
    def version(self) -> str:
        """
        Defines the version of the data in the dataset. When the data is static you can return a static string.
        :return: The version number of the dataset.
        """
        return self._version
