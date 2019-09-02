import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND
from babilim.annotations import RunOnlyOnce
from babilim.layers import IModel, register_layer
from babilim.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Linear, ReLU


@register_layer(TF_BACKEND, "FashionMnistModel")
@register_layer(PYTORCH_BACKEND, "FashionMnistModel")
class FashionMnistModel(IModel):
    def __init__(self, name="FashionMnistModel"):
        super().__init__(name, layer_type="FashionMnistModel")
        l2_weight = 0.2
        out_features = 10
        self.linear = []
        
        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=12, kernel_size=(3, 3), kernel_l2_weight=l2_weight))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3), kernel_l2_weight=l2_weight))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3), kernel_l2_weight=l2_weight))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3), kernel_l2_weight=l2_weight))
        self.linear.append(ReLU())
        self.linear.append(GlobalAveragePooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Linear(out_features=out_features))

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        net = features
        for l in self.linear:
            net = l(net)
        return net
