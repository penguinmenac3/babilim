from typing import Tuple, Iterable

import babilim
import babilim.logger as logging
import babilim.optimizers.learning_rates as lr

from babilim import PYTORCH_BACKEND, PHASE_TRAIN, PHASE_VALIDATION
from babilim.core import RunOnlyOnce, ITensor, Tensor
from babilim.data import Dataset, image_grid_wrap
from babilim.experiment import Config
from babilim.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, ReLU, Flatten
from babilim.losses import Loss, Metrics, SparseCrossEntropyLossFromLogits, SparseCategoricalAccuracy
from babilim.models import IModel

# Use torch optimizer and Linear layer for example
from torch.nn import Linear
from torch.optim import SGD
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


class FashionMnistModel(IModel):
    def __init__(self, config: FashionMnistConfig, name: str = "FashionMnistModel"):
        super().__init__(name, layer_type="FashionMnistModel")
        # Store config so it is availible in build.
        self.config = config

        # Babilim Layers should be initialized in __init__ but could also be initialized in build. (Both would work)
        self.linear = []
        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=12, kernel_size=(3, 3)))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3)))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3)))
        self.linear.append(ReLU())
        self.linear.append(MaxPooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Conv2D(filters=18, kernel_size=(3, 3)))
        self.linear.append(ReLU())
        self.linear.append(GlobalAveragePooling2D())

        self.linear.append(BatchNormalization())
        self.linear.append(Flatten())

    @RunOnlyOnce
    def build(self, features: ITensor):
        # I am lazy, so I just forward pass the features to know what input shape the linear unit has.
        net = features
        for l in self.linear:
            net = l(net)
        num_feats = net.shape[-1]
        
        # now that the input features are known create the remainder of the network.
        self.l1 = Linear(in_features=num_feats,out_features=18)
        self.relu = ReLU()
        self.l2 = Linear(in_features=18, out_features=self.config.problem_number_of_categories)

    def call(self, features: ITensor) -> NetworkOutput:
        babilim_tensor = features
        for l in self.linear:
            babilim_tensor = l(babilim_tensor)
        pytorch_tensor = babilim_tensor.native
        pytorch_tensor = self.l1(pytorch_tensor)
        pytorch_tensor = self.relu(pytorch_tensor)
        pytorch_tensor = self.l2(pytorch_tensor)
        babilim_tensor = Tensor(data=pytorch_tensor, trainable=True)
        return NetworkOutput(class_id=babilim_tensor)


class FashionMnistLoss(Loss):
    def __init__(self):
        super().__init__()
        self.ce = SparseCrossEntropyLossFromLogits()

    def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> ITensor:
        #tprint("y_pred={} y_true={}".format(y_pred.class_id.shape, y_true.class_id.shape))
        return self.ce(y_pred.class_id, y_true.class_id).mean()


class FashionMnistMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.ce = SparseCrossEntropyLossFromLogits()
        self.ca = SparseCategoricalAccuracy()

    def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> None:
        self.log("ce", self.ce(y_pred.class_id, y_true.class_id).mean())
        self.log("ca", self.ca(y_pred.class_id, y_true.class_id).mean())


class MySGD(object):
    def __init__(self, model: IModel, momentum: float=0.95, dampening: float=0.00, weight_decay: float=0, nesterov: bool=True):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.model = model

    @RunOnlyOnce
    def build(self, lr):
        self.sgd = SGD(self.model.trainable_variables_native, lr=lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay, nesterov=self.nesterov)

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor], learning_rate: float) -> None:
        self.build(learning_rate)
        for param_group in self.sgd.param_groups:
            param_group['lr'] = learning_rate
        self.sgd.step()


if __name__ == "__main__":
    babilim.set_backend(PYTORCH_BACKEND)

    # Create our configuration (containing all hyperparameters)
    config = FashionMnistConfig()
    logging.setup(config)

    # Load the data
    train = FashionMnistDataset(config, PHASE_TRAIN)
    val = FashionMnistDataset(config, PHASE_VALIDATION)

    # Create a model.
    model = FashionMnistModel(config)

    # Create a loss and some metrics (if your loss has hyperparameters use config for that)
    loss = FashionMnistLoss()
    metrics = FashionMnistMetrics()

    # Create optimizer
    optim = MySGD(model)

    # Fit our model to the data using our loss and report the metrics.
    model.fit(train, val, loss, metrics, config, optim, config.train_learning_rate_shedule, verbose=True)
