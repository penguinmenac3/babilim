from typing import Tuple
import babilim
import babilim.experiment.logging as logging
from babilim.experiment.logging import tprint
from babilim import PYTORCH_BACKEND, TF_BACKEND, PHASE_TRAIN, PHASE_TEST, PHASE_VALIDATION
from babilim.annotations import RunOnlyOnce
from babilim.models import IModel
from babilim.core.itensor import ITensor
from babilim.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Linear, ReLU, Flatten
from babilim.data import Dataset
from babilim.experiment import Config
from babilim.optimizers import SGD
import babilim.optimizers.learning_rates as lr
from babilim.losses import Loss, Metrics, SparseCrossEntropyLossFromLogits, MeanSquaredError, SparseCategoricalAccuracy
from babilim.core.image_grid import image_grid_wrap

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
        #self.problem_tf_records_path = "tfrecords"

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
        if babilim.is_backend(TF_BACKEND):
            from tensorflow.keras.datasets import fashion_mnist
            ((trainX, trainY), (valX, valY)) = fashion_mnist.load_data()
            self.trainX = trainX
            self.trainY = trainY
            self.valX = valX
            self.valY = valY
        else:
            from torchvision.datasets import FashionMNIST
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
        l2_weight = config.train_l2_weight
        out_features = config.problem_number_of_categories
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
        self.linear.append(Linear(out_features=18))
        self.linear.append(ReLU())
        self.linear.append(Linear(out_features=out_features))

    @RunOnlyOnce
    def build(self, features: ITensor):
        pass

    def call(self, features: ITensor) -> NetworkOutput:
        net = features
        for l in self.linear:
            #print(l.name)
            net = l(net)
            #print(net.shape)
        return NetworkOutput(class_id=net)


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


if __name__ == "__main__":
    babilim.set_backend(PYTORCH_BACKEND)

    # Create our configuration (containing all hyperparameters)
    config = FashionMnistConfig()
    logging.setup(config, continue_training=True)

    # Load the data
    train = FashionMnistDataset(config, PHASE_TRAIN)
    val = FashionMnistDataset(config, PHASE_VALIDATION)

    # Create a model.
    model = FashionMnistModel(config)

    # Create a loss and some metrics (if your loss has hyperparameters use config for that)
    loss = FashionMnistLoss()
    metrics = FashionMnistMetrics()

    # Create optimizer
    optim = SGD()

    # Fit our model to the data using our loss and report the metrics.
    model.fit(train, val, loss, metrics, config, optim, config.train_learning_rate_shedule, verbose=True)
