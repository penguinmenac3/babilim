from torch.utils.data import Dataset as __TDataset
from torch.utils.data import DataLoader as __DataLoader
from typing import Sequence
from babilim.experiment import Config
import numpy as np


class _PyTorchDataset(__TDataset):
    """
    Converts a dataset into a pytorch dataset.

    :param dataset: The dataset to be wrapped.
    """
    def __init__(self, dataset: Sequence):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        feat, label = self.dataset[idx]
        out_f = []
        out_l = []
        for f in feat:
            out_f.append(np.swapaxes(f, 0, -1))
        for l in label:
            if len(l.shape) > 1:
                out_l.append(np.swapaxes(l, 0, -1))
            else:
                out_l.append(l)
        return type(feat)(*out_f), type(label)(*out_l)


def BatchedPytorchDataset(dataset: Sequence, config: Config, shuffle: bool = True, num_workers: int = 1) -> __DataLoader:
    """
    Converts a dataset into a pytorch dataloader.

    :param dataset: The dataset to be wrapped. Only needs to implement list interface.
    :param shuffle: If the data should be shuffled.
    :param num_workers: The number of workers used for preloading.
    :return: A pytorch dataloader object.
    """
    return __DataLoader(_PyTorchDataset(dataset), batch_size=config.train_batch_size,
                        shuffle=shuffle, num_workers=num_workers)
