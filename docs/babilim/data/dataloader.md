[Back to Overview](../../README.md)

# babilim.data.dataloader

> A dataloader object loads the data to the gpu for training.

# *class* **Dataloader**(Iterable)

The dataloader is a wrapper around native dataloaders.

This API ensures that the data is on the GPU in babilim tensors and in a named tuple.

You can iterate over the dataloader to get training samples.
To get information about the original dataset you can use `self.dataset`.

* native_dataloader: The native dataloader, that should be wrapped.
* dataset: The original babilim dataset to allow a user getting information about it, if required.


### *class* **TensorDataloaderIterator**(Iterator)

*(no documentation found)*

