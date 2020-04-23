from babilim.model.module import Module


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__(layer_type="Sequential")
        self.layers = layers

    def call(self, features):
        tmp = features
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp