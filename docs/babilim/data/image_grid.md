[Back to Overview](../../README.md)

# babilim.data.image_grid

> Wrap an image tensor so that it is in the right format for tensorflow or pytorch.

### *def* **image_grid_wrap**(data: np.ndarray) -> np.ndarray

Prepares 2D grid information to be used in your neural network.
This is required for all data that should be usable by a 2D Convolution.

If your data has shape [H, W] it gets transformed to [H, W, 1] automatically.
For pytorch the data gets further reordered in a channel first order [C, H, W].

* data: The data that should be prepared.
* returns: A numpy ndarray with the prepared image/grid data.


### *def* **image_grid_unwrap**(data: np.ndarray) -> np.ndarray

For pytorch the data gets reordered in a channel last order [H, W, C].

* data: The data that should be unwrapped.
* returns: A numpy ndarray with the unwrapped image/grid data.


