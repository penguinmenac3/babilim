import babilim
import numpy as np

def image_grid_wrap(data: np.ndarray) -> np.ndarray:
    """
    Prepares 2D grid information to be used in your neural network.
    This is required for all data that should be usable by a 2D Convolution.

    If your data has shape [H, W] it gets transformed to [H, W, 1] automatically.
    For pytorch the data gets further reordered in a channel first order [C, H, W].
    
    Arguments:
        data {np.ndarray} -- The data that should be prepared.
    
    Returns:
        np.ndarray -- A numpy ndarray with the prepared image/grid data.
    """
    if data.ndim != 3 and data.ndim != 2:
        raise ValueError("Wrong dimensionality of the data. Must be 2/3 dimensional but is {} dimensional.".format(data.ndim))

    # Add a dimension at the end.
    if data.ndim == 2:
        data = data[:, :, None]

    # Transpose data for numpy.
    if babilim.is_backend(babilim.PYTORCH_BACKEND):
        return data.transpose((2, 0, 1))
    else:
        return data


def image_grid_unwrap(data: np.ndarray) -> np.ndarray:
    """
    For pytorch the data gets reordered in a channel last order [H, W, C].
    
    Arguments:
        data {np.ndarray} -- The data that should be unwrapped.
    
    Returns:
        np.ndarray -- A numpy ndarray with the unwrapped image/grid data.
    """
    if data.ndim != 3:
        raise ValueError("Wrong dimensionality of the data. Must be 3 dimensional but is {} dimensional.".format(data.ndim))
    # Transpose data for numpy.
    if babilim.is_backend(babilim.PYTORCH_BACKEND):
        return data.transpose((1, 2, 0))
    else:
        return data
