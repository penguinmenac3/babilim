from babilim import PYTORCH_BACKEND, TF_BACKEND, is_backend, get_backend
from babilim.layers.ilayer import ILayer, register_layer

@register_layer(TF_BACKEND, "Linear")
@register_layer(PYTORCH_BACKEND, "Linear")
def Linear(out_features: int, name:str ="Linear") -> ILayer:
    """A simple linear layer.

    It computes Wx+b with no activation funciton.
    
    Arguments:
        out_features {int} -- The number of output features.
    
    Keyword Arguments:
        name {str} -- The name of your layer. (default: {"Linear"})
    
    Raises:
        NotImplementedError: When an unsupported backend is set. PYTORCH_BACKEND and TF_BACKEND are supported.
    
    Returns:
        ILayer -- A layer object.
    """
    if is_backend(PYTORCH_BACKEND):
        from babilim.layers.pt.linear import Linear as _Linear
        return _Linear(out_features, name)
    elif is_backend(TF_BACKEND):
        from babilim.layers.tf.linear import Linear as _Linear
        return _Linear(out_features, name)
    else:
        raise NotImplementedError("The backend {} is not implemented by this layer.".format(get_backend()))
