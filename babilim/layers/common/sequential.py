from babilim.layers.ilayer import ILayer, register_layer
from babilim import PYTORCH_BACKEND, TF_BACKEND


@register_layer(TF_BACKEND, "Sequential")
@register_layer(PYTORCH_BACKEND, "Sequential")
class Sequential(ILayer):
    def __init__(self, name, *layers):
        super().__init__(name=name, layer_type="Sequential")
        self.layers = layers

    def call(self, features):
        tmp = features
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp
