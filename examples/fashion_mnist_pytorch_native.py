from typing import Tuple, Iterable

import babilim
import babilim.logger as logger
import babilim.optimizers.learning_rates as lr

from babilim import PYTORCH_BACKEND, PHASE_TRAIN, PHASE_VALIDATION
from babilim.core import RunOnlyOnce
from babilim.data import Dataset, image_grid_wrap
from babilim.experiment import Config
from babilim.losses import NativeMetricsWrapper, NativeLossWrapper
from babilim.models import NativeModelWrapper
from babilim.optimizers import NativePytorchOptimizerWrapper

import torch
from torch import Tensor
from torch.nn import Linear, Module, BatchNorm2d, Conv2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision.datasets import FashionMNIST

import numpy as np
from collections import namedtuple


# Create some named tuple for our inputs and outputs so we do not confuse them.
NetworkInput = namedtuple("NetworkInput", ["features"])
NetworkOutput = namedtuple("NetworkOutput", ["class_id"])


class FashionMnistConfig(Config):
    def __init__(self):
        super().__init__()
        self.problem_number_of_categories = 10
        self.problem_samples = 1875 * 32
        self.problem_base_dir = "datasets"

        self.train_epochs = 20
        self.train_l2_weight = 0.01
        self.train_batch_size = 32
        self.train_log_steps = 100
        self.train_experiment_name = "FashionMNIST"
        self.train_checkpoint_path = "checkpoints"
        samples_per_epoch = self.problem_samples / self.train_batch_size
        self.train_learning_rate_shedule = lr.Exponential(initial_lr=0.001, k=0.1 / samples_per_epoch)


class FashionMnistDataset(Dataset):
    def __init__(self, config: FashionMnistConfig, phase: str):
        super().__init__(config)
        dataset = FashionMNIST(config.problem_base_dir, train=phase==PHASE_TRAIN, download=True)
        self.trainX = []
        self.trainY = []
        for x, y in dataset:
            self.trainX.append(x)
            self.trainY.append(y)
        self.valX = self.trainX
        self.valY = self.trainY
        self.training = phase == PHASE_TRAIN

    def __len__(self) -> int:
        if self.training:
            return int(len(self.trainX))
        else:
            return int(len(self.valX))

    def getitem(self, idx: int) -> Tuple[NetworkInput, NetworkOutput]:
        if self.training:
            label = np.array(self.trainY[idx], dtype="uint8")
            feat = np.array(self.trainX[idx], dtype="float32")
        else:
            label = np.array(self.valY[idx], dtype="uint8")
            feat = np.array(self.valX[idx], dtype="float32")

        feat = np.reshape(feat, (28, 28))
        return NetworkInput(features=image_grid_wrap(feat)), NetworkOutput(class_id=label)

    @property
    def version(self) -> str:
        return "FashionMnistDataset"


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


class FashionMnistLoss(Module):
    def __init__(self):
        super().__init__()
        self.ce = CrossEntropyLoss()

    def forward(self, y_pred: NetworkOutput, y_true: NetworkOutput, log_val) -> Tensor:
        return self.ce(y_pred.class_id, y_true.class_id.long()).mean()


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


if __name__ == "__main__":
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
