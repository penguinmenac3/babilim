# babilim.training.trainer

> Every trainer implements the trainer interface.

# *class* **Trainer**(object)

A trainer is a general interface for training models.

**You must not not instantiate this directly, instead use implementations of trainers.**


### *def* **run_epoch**(*self*, dataloader: Dataloader, phase: str, epoch: int)

Run an epoch in training or validation.

(This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

* dataloader: The dataloader created from a dataset.
* phase: The phase (train/dev/test) which is used for running.
* epoch: The epoch number.
* returns: Returns the average loss.


### *def* **fit**(*self*, train_dataloader: Dataloader, dev_dataloader: Dataloader, epochs: int)

Fit the model managed by this trainer to the data.

* train_dataloader: The dataloader for training your neural network (train split).
* dev_dataloader: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
* epochs: The number of epochs describes how often the fit will iterate over the dataloaders.


