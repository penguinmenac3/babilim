MNIST Example (Pytorch Native)
==============================

This example shows how to implement a simple network solving MNIST with babilim but with the network, loss, metrics  and optimizer written in pytorch.
The focus is not to create the shortest possible solution, but rather to show how babilim works.
The example is build in a way it can be applied to other problems as well.

The full code can be found `here on github <https://github.com/penguinmenac3/babilim/blob/master/examples/fashion_mnist_pytorch_native.py>`_.

**1.-4. Follow instructions in MNIST Example (Babilim)**

These steps are the same as in the MNIST Example (Babilim).
Follow them and come back to this tutorial afterwards.

**5. Implementing the model**

TODO

.. code-block:: python

    from torch.nn import Linear, Module, BatchNorm2d, Conv2d
    from torch.nn.functional import relu, max_pool2d, avg_pool2d

    class FashionMnistModel(Module):
        def __init__(self, config: FashionMnistConfig):
            super().__init__()
            self.config = config
            self.layers = []

        def register(self, layer):
            if torch.cuda.is_available():
                layer = layer.to(torch.device("cuda"))
            self.__setattr__("layer_{}".format(len(self.layers)), layer)
            return layer

        def make_bn(self, features):
            self.layers.append(self.register(BatchNorm2d(features.shape[1])))
            return self.layers[-1](features)

        def make_conv2d(self, features, filters, kernel_size):
            px = int((kernel_size[0] - 1) / 2)
            py = int((kernel_size[1] - 1) / 2)
            padding = (px, py)
            self.layers.append(self.register(Conv2d(features.shape[1], filters, kernel_size, (1, 1), padding)))
            return self.layers[-1](features)

        def make_relu(self, features):
            self.layers.append(relu)
            return self.layers[-1](features)

        def make_max_pool_2d(self, features):
            self.layers.append(lambda x: max_pool2d(x, (2, 2)))
            return self.layers[-1](features)

        def make_global_avg_pool_2d(self, features):
            self.layers.append(lambda x: avg_pool2d(x, features.size()[2:]))
            return self.layers[-1](features)

        def make_flatten(self, features):
            self.layers.append(lambda x: x.view(x.shape[0], -1))
            return self.layers[-1](features)

        def make_linear(self, net, out_features):
            self.layers.append(self.register(Linear(in_features=net.shape[-1], out_features=out_features)))
            return self.layers[-1](net)

        @RunOnlyOnce
        def build(self, features):
            net = features
            net = self.make_bn(net)
            net = self.make_conv2d(net, 12, (3, 3))
            net = self.make_relu(net)
            net = self.make_max_pool_2d(net)
            net = self.make_bn(net)
            net = self.make_conv2d(net, 18, (3, 3))
            net = self.make_relu(net)
            net = self.make_max_pool_2d(net)
            net = self.make_bn(net)
            net = self.make_conv2d(net, 18, (3, 3))
            net = self.make_relu(net)
            net = self.make_max_pool_2d(net)
            net = self.make_bn(net)
            net = self.make_conv2d(net, 18, (3, 3))
            net = self.make_relu(net)
            net = self.make_global_avg_pool_2d(net)
            net = self.make_bn(net)
            net = self.make_flatten(net)
            net = self.make_linear(net, 18)
            net = self.make_relu(net)
            net = self.make_linear(net, self.config.problem_number_of_categories)

        def forward(self, features) -> NetworkOutput:
            tensor = features
            for l in self.layers:
                tensor = l(tensor)
            return NetworkOutput(class_id=tensor)

**6. Defining the Loss and Metrics**

With a model, the last step before training is to setup some losses and metrics.

The loss is pretty simple. It is a class implementing a call function which has three parameters.
The first parameter is y_pred representing the actual network output, and y_true is the intended network output as returned by the dataset.
The last parameter log_val(name, tensor) is a function that can be used to log intermediate computations such as partial losses.
The return type of the loss is a single Tensor, the loss that should be optimized.


.. code-block:: python

    from torch.nn import Module
    from torch.nn import CrossEntropyLoss

    class FashionMnistLoss(Module):
        def __init__(self):
            super().__init__()
            self.ce = CrossEntropyLoss()

        def forward(self, y_pred: NetworkOutput, y_true: NetworkOutput, log_val) -> Tensor:
            return self.ce(y_pred.class_id, y_true.class_id.long()).mean()


The metric is very similar to the loss. It is a class implementing a call function which has three parameters.
The first parameter is y_pred representing the actual network output, and y_true is the intended network output as returned by the dataset.
The last parameter log_val(name, tensor) is a function that can be used to log the computed metrics.
However, it does not have a return type and no effect on the optimization.

.. code-block:: python

    from torch import Tensor
    from torch.nn import Module
    from torch.nn import CrossEntropyLoss

    class FashionMnistMetrics(Module):
        def __init__(self):
            super().__init__()
            self.ce = CrossEntropyLoss()

        def ca(self, y_pred: Tensor, y_true: Tensor):
            pred_class = y_pred.argmax(dim=-1)
            true_class = y_true.long()
            correct_predictions = pred_class == true_class
            return correct_predictions.float().mean()

        def forward(self, y_pred: NetworkOutput, y_true: NetworkOutput, log_val) -> None:
            log_val("ce", self.ce(y_pred.class_id, y_true.class_id.long()).mean())
            log_val("ca", self.ca(y_pred.class_id, y_true.class_id).mean())

**7. Training it**

TODO

.. code-block:: python

    import babilim.logger as logger
    from babilim import PYTORCH_BACKEND, PHASE_TRAIN, PHASE_VALIDATION
    from babilim.losses import NativeMetricsWrapper, NativeLossWrapper
    from babilim.models import NativeModelWrapper
    from babilim.optimizers import NativePytorchOptimizerWrapper

    from torch.optim import SGD

    babilim.set_backend(PYTORCH_BACKEND)

    # Create our configuration (containing all hyper parameters)
    config = FashionMnistConfig()
    logger.setup(config, continue_training=False)

    # Load the data
    train = FashionMnistDataset(config, PHASE_TRAIN)
    val = FashionMnistDataset(config, PHASE_VALIDATION)

    # Create a model.
    model = NativeModelWrapper(FashionMnistModel(config), name="FashionMnistModel")

    # Create a loss and some metrics (if your loss has hyper parameters use config for that)
    loss = NativeLossWrapper(FashionMnistLoss())
    metrics = NativeMetricsWrapper(FashionMnistMetrics())

    # Create optimizer
    optim = NativePytorchOptimizerWrapper(SGD, model, momentum=0.95, dampening=0.0, weight_decay=0.0, nesterov=True)

    # Fit our model to the data using our loss and report the metrics.
    model.fit(train, val, loss, metrics, config, optim, config.train_learning_rate_shedule, verbose=True)

**8. What next?**

Solve your own problem in a similar manner.
Dive into the detailed api documentation and even have peeks at the code to become a true master in using babilim.
