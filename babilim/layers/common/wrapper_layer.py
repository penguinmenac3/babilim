from typing import Callable
from babilim.core.tensor import Tensor
from babilim.layers.ilayer import ILayer, register_layer
from babilim import PYTORCH_BACKEND, TF_BACKEND


@register_layer(TF_BACKEND, "Lambda")
@register_layer(PYTORCH_BACKEND, "Lambda")
class Lambda(ILayer):
    def __init__(self, name: str, native_layer: Callable):
        """
        Wrap a native function in a layer.
        
        :param name: The name of the function.
        :param native_layer: The native layer or function to wrap. (Must accept *args and or **kwargs and return a single tensor, a list of tensors or a dict of tensors)
        """
        super().__init__(name=name, layer_type="Lambda")
        self.native_layer = native_layer

    def call(self, *args, **kwargs):
        # Unwrap arguments
        args = [feature.native for feature in args]
        kwargs = {k: kwargs[k].native for k in kwargs}
        
        # call function
        result = self.native_layer(*args, **kwargs)

        # Wrap results
        if isinstance(result, dict):
            result = {k: Tensor(data=result[k], trainable=True) for k in result}
        elif isinstance(result, list):
            result = [Tensor(data=res, trainable=True) for res in result]
        else:
            result = Tensor(data=result, trainable=True)
        return result
