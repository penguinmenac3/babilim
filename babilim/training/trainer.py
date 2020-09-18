# AUTOGENERATED FROM: babilim/training/trainer.ipynb

# Cell: 0
"""doc
# babilim.training.trainer

> Every trainer implements the trainer interface.
"""

# Cell: 1
from babilim.data import Dataloader


# Cell: 2
class Trainer(object):
    def __init__(self):
        """
        A trainer is a general interface for training models.
        
        **You must not not instantiate this directly, instead use implementations of trainers.**
        """
        raise RuntimeError("You must not instantiate this class directly.")
    
    def run_epoch(self, dataloader: Dataloader, phase: str, epoch: int):
        """
        Run an epoch in training or validation.

        (This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

        Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

        :param dataloader: The dataloader created from a dataset.
        :param phase: The phase (train/dev/test) which is used for running.
        :param epoch: The epoch number.
        :return: Returns the average loss.
        """
        raise NotImplementedError()

    def fit(self, train_dataloader: Dataloader, dev_dataloader: Dataloader, epochs: int):
        """
        Fit the model managed by this trainer to the data.

        :param train_dataloader: The dataloader for training your neural network (train split).
        :param dev_dataloader: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
        :param epochs: The number of epochs describes how often the fit will iterate over the dataloaders.
        """
        raise NotImplementedError()
