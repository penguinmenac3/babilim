[Back to Overview](../README.md)

# Example: MNIST with Babilim (Mixing with Native Pytorch)

> A babilim example to solve fashion MNIST with a gpu.

Before we start with the solving, make sure you have installed babilim and pytorch (or tf2).

```bash
conda activate my_env_name
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install babilim
```

Then to start, we setup babilim to use the backend of choice in our notebook and set the debug verbosity high.

Example:
```python
import babilim

babilim.DEBUG_VERBOSITY = True
babilim.set_backend(babilim.PYTORCH_BACKEND)
```
Output:
```
[2020-05-14 15:59:35] INFO Using backend: pytorch-gpu

```

## Defining the Problem

MNIST contains images of numbers 0-9.
The images are grayscale and have a resolution of 28x28.
For every image the number that has been written is annotated as an integer number.

We want to build a neural network that predicts the number given the image.
Typically the input to the neural network can be called features or image (we chose features, since it is more generic here).
The output in this case can be seen as a classification problem.
We have classes from 0  to 9 which are mutually exclusive, since a prediction can be only one number.
However classes should be prefered over regression, since they can model uncertainty between 8 and 2 better by assigning both high probabilities, whereas regression would yield 5 (which is completely wrong).

Those decisions lead to a problem definition as follows:

$
\texttt{features} \rightarrow \texttt{class-id}
$

Example:
```python
# e.g.: definitions/mnist.py
from collections import namedtuple

# Create some named tuple for our inputs and outputs so we do not confuse them.
NetworkInput = namedtuple("NetworkInput", ["features"])
NetworkOutput = namedtuple("NetworkOutput", ["class_id"])
```

## Defining a configuration

Whereas we anticipate running multiple experiments over the course of the development, we want to have all configuration choices in one config class, so we can easily track the experiments.

At the beginning this class will be created empty and then filled with life over time.
In this example it will directly come with all variables required, since this cell has been edited multiple times during the development process.



Example:
```python
# e.g.: configs/mnist.py
from babilim.core import Config
from babilim.training.optimizers import learning_rates


class FashionMnistConfig(Config):
    def __init__(self):
        super().__init__()
        self.problem_number_of_categories = 10
        self.problem_samples = 60000
        self.problem_base_dir = "datasets"

        self.train_epochs = 20
        self.train_l2_weight = 0.01
        self.train_batch_size = 32
        self.train_log_steps = 100
        self.train_experiment_name = "FashionMNIST"
        self.train_checkpoint_path = "checkpoints"
        samples_per_epoch = self.problem_samples / self.train_batch_size
        self.train_learning_rate_shedule = learning_rates.Exponential(initial_lr=0.001, k=0.1 / samples_per_epoch)


# Create a config (should be in your main function)
config = FashionMnistConfig()
```

## Trinity of Dataset, Model, Trainer

In every deep learning problem there is the trinity of the dataset, the model and the trainer.
If any of the parts is weak, the overall result will be unsatisfactory.

This means we want to ensure that all parts work equally good.
Without good data, a good model is of no use at all.
So  we will start with the dataset and explore it.

This is typically a good way to start, whereas model choice is dependant on the data.

## Dataset

In the case of MNIST we are in a comfortable spot, whereas the dataset is already balanced, and nicely preprocessed.
All images are centered, have the same size and have good contrast.
However, we will use a transformer which does not change the data, to show how a tranformer would work.

The dataset can be easily implemented using tensorflow and pytorch libraries, whereas they provide loaders for the data already.
This gives us a chance to show writing specific code for a backend.

Example:
```python
# e.g.: datasets/mnist.py
import babilim
import numpy as np
from babilim import PHASE_TRAIN
from babilim.data import Dataset
from typing import Tuple

from torchvision.datasets import FashionMNIST as _FashionMNISTDataset


class FashionMnistDataset(Dataset):
    def __init__(self, config: FashionMnistConfig, phase: str):
        super().__init__(config)
        self.training = phase == PHASE_TRAIN
        dataset = _FashionMNISTDataset(config.problem_base_dir, train=phase==PHASE_TRAIN, download=True)
        self.inp = []
        self.outp = []
        for x, y in dataset:
            self.inp.append(x)
            self.outp.append(y)

    def __len__(self) -> int:
        return int(len(self.inp))

    def getitem(self, idx: int) -> Tuple[NetworkInput, NetworkOutput]:
        feat = np.array(self.inp[idx], dtype="float32")
        label = np.array(self.outp[idx], dtype="uint8")
        
        feat = np.reshape(feat, (28, 28))
        return NetworkInput(features=feat), NetworkOutput(class_id=label)

    @property
    def version(self) -> str:
        return "FashionMnistDataset"
```

Now we will write a simple transformer that makes the data readable for the neural network training.

The transformer will make an image grid out of the feature, so it can be used in tensorflow or pytorch, since one has channel first and the other channel last representation.

Example:
```python
# e.g.: datasets/mnist_transformers.py
from babilim.data import Transformer, image_grid_wrap
from typing import Tuple


class MNISTTransformer(Transformer):
    def __init__(self):
        pass
    
    def __call__(self, inp: NetworkInput, outp: NetworkOutput) -> Tuple[NetworkInput, NetworkOutput]:
        return NetworkInput(features=image_grid_wrap(inp.features)), outp
    
    @property
    def version(self):
        return "MNISTTransformer"
```

The dataset is created and then a transformer is appended to the transformers.
There are two types of transformers: `dataset::transformers` and `dataset::realtime_transformers`.
The difference between the two is that the realtime_transformers get applied after caching and the transformers before caching.

> Important: Realtime transformers should contain easy and quick computations to avoid slowing down your training and data augmentation should be done in realtime transformers or it will only be applied once (before caching).

Example:
```python
# Create the dataset and setup the transformers (should be in your main function)
from babilim import PHASE_VALIDATION, PHASE_TRAIN


train = FashionMnistDataset(config, PHASE_TRAIN)
val = FashionMnistDataset(config, PHASE_VALIDATION)
transformer = MNISTTransformer()
train.realtime_transformers.append(transformer)
val.realtime_transformers.append(transformer)
```
Output:
```
  1%|          | 147456/26421880 [00:00<00:17, 1474231.01it/s]Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz
26427392it [00:01, 18935082.67it/s]                              
Extracting datasets/FashionMNIST/raw/train-images-idx3-ubyte.gz
32768it [00:00, 421150.06it/s]
 12%|█▏        | 516096/4422102 [00:00<00:00, 4808258.42it/s]Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting datasets/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
4423680it [00:00, 14259136.02it/s]                            
8192it [00:00, 165470.28it/s]
Extracting datasets/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Processing...
Done!

```

One last step is to test if the dataset works.
To test this we will call the dataset with multiple random indices.
(The images you will see when executing might differ.)
Normally this test would be much more detailed, checking the value ranges of your data, if everything is aligned and as expected, but in this simple example we will only do a visual inspection.

Example:
```python
# e.g. tests/mnist_visualization.py
import matplotlib.pyplot as plt
import random
from babilim.data import image_grid_unwrap

print("Training")
plt.figure(figsize=(16,6))
for i in range(24):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        plt.subplot(3, 8, i + 1)
        plt.axis('off')
        plt.title(label.class_id)
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()

print("Validation")
plt.figure(figsize=(16,6))
for i in range(24):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        plt.subplot(3, 8, i + 1)
        plt.axis('off')
        plt.title(label.class_id)
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()
```
Output:
```
Training
Validation

```
![data](../../docs/jlabdev_images/f31c46b7a5e12d7c9cfb38669da841d6.png)
![data](../../docs/jlabdev_images/418498c9a0ad7e4db558a09c99190773.png)

## The Model

After having implemented and tested the dataloader.
We can move to the next step: Implementing the model.

This will actually be a suprisingly small part of this tutorial, since it is so easy with modern frameworks.

### CNN

The problem is classifying images into the clothing type, therefore a convolutional neural network is the right model choice.
If you do not now what a CNN is, please refer to one of the plenty resources available on the internet.

We will use a CNN inspired by VGG 16.
So it wil have convolutions followed by a few fully connected layers.

### Implementation

The model has an `__init__`, a `build`, and a `call`-function that need to be implemented.

The `build`-function with the `@RunOnlyOnce`-annotation will be run only once and build the model initially.
It has available knowledge on the shape of the input tensors.

The `call`-function is the actual forward pass of the model and typically trivial to implement when using the build-call pattern.

Example:
```python
# e.g. model/mnist.py
from babilim.core import RunOnlyOnce
from torch.nn import Module
from torch import Tensor
from torch.nn.functional import max_pool2d, relu
from babilim.model.modules import BatchNormalization, Conv2D, GlobalAveragePooling2D, Linear, Flatten
from babilim.model.modules import Lambda

class FashionMnistModel(Module):
    def __init__(self, config: FashionMnistConfig):
        super().__init__()
        self.config = config
        
    @RunOnlyOnce
    def build(self, features: Tensor):
        out_features = self.config.problem_number_of_categories
        # Create all layers. Mix and match torch and babilim as you like.
        # I prefer babilim, but torch.nn.conv2d would also work.
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters=12, kernel_size=(3, 3))
        
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(filters=18, kernel_size=(3, 3))
        
        self.bn3 = BatchNormalization()
        self.conv3 = Conv2D(filters=18, kernel_size=(3, 3))
        
        self.bn4 = BatchNormalization()
        self.conv4 = Conv2D(filters=18, kernel_size=(3, 3))
        
        self.bn5 = BatchNormalization()
        self.conv5 = Conv2D(filters=18, kernel_size=(3, 3))
        self.global_pooling = GlobalAveragePooling2D()
        
        self.bn6 = BatchNormalization()
        self.flatten = Flatten()
        self.fc1 = Linear(out_features=18)
        self.fc2 = Linear(out_features=out_features)

    def forward(self, features: Tensor) -> NetworkOutput:
        net = features
        net = self.bn1(net)
        net = self.conv1(net)
        net = relu(net)
        net = max_pool2d(net, (2, 2))
        
        net = self.bn2(net)
        net = self.conv2(net)
        net = relu(net)
        net = max_pool2d(net, (2, 2))
        
        
        net = self.bn3(net)
        net = self.conv3(net)
        net = relu(net)
        net = max_pool2d(net, (2, 2))
        
        net = self.bn4(net)
        net = self.conv4(net)
        net = relu(net)
        net = max_pool2d(net, (2, 2))
        
        net = self.bn5(net)
        net = self.conv5(net)
        net = relu(net)
        net = max_pool2d(net, (2, 2))
        net = self.global_pooling(net)
        
        net = self.bn6(net)
        net = self.flatten(net)
        net = self.fc1(net)
        net = relu(net)
        net = self.fc2(net)
        
        return NetworkOutput(class_id=net)


# Create the model and initialize it (should be in your main function)
model = FashionMnistModel(config)
model = Lambda(model)  # Make nn.Module a babilim.Module
model.initialize(train)
```
Output:
```
[2020-05-14 15:59:54] INFO Build Model

```

Testing the model is important to see if it works and transforms the data as expected.
The output will not be correct but should appear random.

Example:
```python
# e.g. test/mnist_model.py
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

model.eval()
print("Training")
plt.figure(figsize=(16,6))
for i in range(12):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        pred = model.predict(**feat._asdict())
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        class_probs = softmax(pred.class_id)
        plt.title(", ".join(["{:.0f}".format(x*100) for x in class_probs]))
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()

print("Validation")
plt.figure(figsize=(16,6))
for i in range(12):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        pred = model.predict(**feat._asdict())
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        class_probs = softmax(pred.class_id)
        plt.title(", ".join(["{:.0f}".format(x*100) for x in class_probs]))
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()
model.train()
```
Output:
```
Training
Validation

```
![data](../../docs/jlabdev_images/520eac670fa6287928884369169298a6.png)
![data](../../docs/jlabdev_images/df357004b0600f37b1f62720c7068a8d.png)

## The Trainer

Having implemented a model, the trainer is the remaining part.
The trainer will fit the model to the dataset and consists of multiple parts.

The loss, metrics, an optimizer and the trainer itself.

### Loss

Example:
```python
# e.g. training/mnist_loss.py
from babilim.training.losses import Loss, SparseCrossEntropyLossFromLogits
from babilim.core import ITensor


class FashionMnistLoss(Loss):
    def __init__(self):
        super().__init__()
        self.ce = SparseCrossEntropyLossFromLogits()

    def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> ITensor:
        return self.ce(y_pred.class_id, y_true.class_id).mean()


# Create the loss (should be in your main function)
loss = FashionMnistLoss()
```

### Metrics

Example:
```python
# e.g. training/mnist_metrics.py
from babilim.training.losses import Metrics, SparseCrossEntropyLossFromLogits, SparseCategoricalAccuracy
from babilim.core import ITensor


class FashionMnistMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self.ce = SparseCrossEntropyLossFromLogits()
        self.ca = SparseCategoricalAccuracy()

    def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> None:
        self.log("ce", self.ce(y_pred.class_id, y_true.class_id).mean())
        self.log("ca", self.ca(y_pred.class_id, y_true.class_id).mean())


# Create the metrics (should be in your main function)        
metrics = FashionMnistMetrics()
```

### Training

Example:
```python
# e.g. training/mnist_train.py (contain all stuff annotated with "should be in your main function" and this cell)
from babilim.core import Config, logger
from babilim.training import supervised
from babilim.training.optimizers import SGD

logger.setup(config, continue_training=False)

# Create optimizer
optim = SGD()

# Fit our model to the data using our loss and report the metrics.
supervised.fit(model, train, val, loss, metrics, config, optim, config.train_learning_rate_shedule, verbose=True)
```
Output:
```
[2020-05-14 16:00:07] INFO Build Model
[2020-05-14 16:00:07] INFO Trainable Variables:
[2020-05-14 16:00:07] INFO   /native_module/bn1/bn/weight: (1,)
[2020-05-14 16:00:07] INFO   /native_module/bn1/bn/bias: (1,)
[2020-05-14 16:00:07] INFO   /native_module/conv1/weight: (12, 1, 3, 3)
[2020-05-14 16:00:07] INFO   /native_module/conv1/bias: (12,)
[2020-05-14 16:00:07] INFO   /native_module/bn2/bn/weight: (12,)
[2020-05-14 16:00:07] INFO   /native_module/bn2/bn/bias: (12,)
[2020-05-14 16:00:07] INFO   /native_module/conv2/weight: (18, 12, 3, 3)
[2020-05-14 16:00:07] INFO   /native_module/conv2/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn3/bn/weight: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn3/bn/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/conv3/weight: (18, 18, 3, 3)
[2020-05-14 16:00:07] INFO   /native_module/conv3/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn4/bn/weight: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn4/bn/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/conv4/weight: (18, 18, 3, 3)
[2020-05-14 16:00:07] INFO   /native_module/conv4/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn5/bn/weight: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn5/bn/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/conv5/weight: (18, 18, 3, 3)
[2020-05-14 16:00:07] INFO   /native_module/conv5/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn6/bn/weight: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn6/bn/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/fc1/weight: (18, 18)
[2020-05-14 16:00:07] INFO   /native_module/fc1/bias: (18,)
[2020-05-14 16:00:07] INFO   /native_module/fc2/weight: (10, 18)
[2020-05-14 16:00:07] INFO   /native_module/fc2/bias: (10,)
[2020-05-14 16:00:07] INFO Untrainable Variables:
[2020-05-14 16:00:07] INFO   /native_module/bn1/bn/running_mean: (1,)
[2020-05-14 16:00:07] INFO   /native_module/bn1/bn/running_var: (1,)
[2020-05-14 16:00:07] INFO   /native_module/bn1/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO   /native_module/bn2/bn/running_mean: (12,)
[2020-05-14 16:00:07] INFO   /native_module/bn2/bn/running_var: (12,)
[2020-05-14 16:00:07] INFO   /native_module/bn2/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO   /native_module/bn3/bn/running_mean: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn3/bn/running_var: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn3/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO   /native_module/bn4/bn/running_mean: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn4/bn/running_var: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn4/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO   /native_module/bn5/bn/running_mean: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn5/bn/running_var: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn5/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO   /native_module/bn6/bn/running_mean: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn6/bn/running_var: (18,)
[2020-05-14 16:00:07] INFO   /native_module/bn6/bn/num_batches_tracked: ()
[2020-05-14 16:00:07] INFO Start training for 20 epochs from epoch 0.
[2020-05-14 16:03:22] STAT Training 1875/1875 - Loss 0.477 - LR 0.000905
[2020-05-14 16:03:54] STAT Validating 312/312 - Loss 0.479
[2020-05-14 16:03:54] STAT Epoch 1/20 - ETA 1:11:53 - loss/total=0.479 - ce=0.479 - ca=0.829
[2020-05-14 16:07:12] STAT Training 1875/1875 - Loss 0.413 - LR 0.000819
[2020-05-14 16:07:44] STAT Validating 312/312 - Loss 0.385
[2020-05-14 16:07:44] STAT Epoch 2/20 - ETA 1:08:34 - loss/total=0.385 - ce=0.385 - ca=0.865
[2020-05-14 16:11:02] STAT Training 1875/1875 - Loss 0.375 - LR 0.000741
[2020-05-14 16:11:34] STAT Validating 312/312 - Loss 0.360
[2020-05-14 16:11:34] STAT Epoch 3/20 - ETA 1:04:51 - loss/total=0.360 - ce=0.360 - ca=0.870
[2020-05-14 16:14:49] STAT Training 1875/1875 - Loss 0.310 - LR 0.000670
[2020-05-14 16:15:20] STAT Validating 312/312 - Loss 0.341
[2020-05-14 16:15:20] STAT Epoch 4/20 - ETA 1:00:51 - loss/total=0.341 - ce=0.341 - ca=0.877
[2020-05-14 16:18:36] STAT Training 1875/1875 - Loss 0.317 - LR 0.000607
[2020-05-14 16:19:09] STAT Validating 312/312 - Loss 0.328
[2020-05-14 16:19:09] STAT Epoch 5/20 - ETA 0:57:03 - loss/total=0.328 - ce=0.328 - ca=0.881
[2020-05-14 16:22:26] STAT Training 1875/1875 - Loss 0.295 - LR 0.000549
[2020-05-14 16:22:59] STAT Validating 312/312 - Loss 0.313
[2020-05-14 16:22:59] STAT Epoch 6/20 - ETA 0:53:20 - loss/total=0.313 - ce=0.313 - ca=0.887
[2020-05-14 16:26:15] STAT Training 1875/1875 - Loss 0.285 - LR 0.000497
[2020-05-14 16:26:47] STAT Validating 312/312 - Loss 0.308
[2020-05-14 16:26:47] STAT Epoch 7/20 - ETA 0:49:30 - loss/total=0.308 - ce=0.308 - ca=0.888
[2020-05-14 16:30:05] STAT Training 1875/1875 - Loss 0.260 - LR 0.000449
[2020-05-14 16:30:37] STAT Validating 312/312 - Loss 0.316
[2020-05-14 16:30:37] STAT Epoch 8/20 - ETA 0:45:44 - loss/total=0.316 - ce=0.316 - ca=0.884
[2020-05-14 16:33:55] STAT Training 1875/1875 - Loss 0.271 - LR 0.000407
[2020-05-14 16:34:27] STAT Validating 312/312 - Loss 0.301
[2020-05-14 16:34:27] STAT Epoch 9/20 - ETA 0:41:57 - loss/total=0.301 - ce=0.301 - ca=0.889
[2020-05-14 16:37:45] STAT Training 1875/1875 - Loss 0.279 - LR 0.000368
[2020-05-14 16:38:17] STAT Validating 312/312 - Loss 0.299
[2020-05-14 16:38:17] STAT Epoch 10/20 - ETA 0:38:10 - loss/total=0.299 - ce=0.299 - ca=0.892
[2020-05-14 16:41:36] STAT Training 1875/1875 - Loss 0.264 - LR 0.000333
[2020-05-14 16:42:08] STAT Validating 312/312 - Loss 0.298
[2020-05-14 16:42:08] STAT Epoch 11/20 - ETA 0:34:22 - loss/total=0.298 - ce=0.298 - ca=0.895
[2020-05-14 16:45:27] STAT Training 1875/1875 - Loss 0.277 - LR 0.000301
[2020-05-14 16:45:59] STAT Validating 312/312 - Loss 0.292
[2020-05-14 16:45:59] STAT Epoch 12/20 - ETA 0:30:34 - loss/total=0.292 - ce=0.292 - ca=0.896
[2020-05-14 16:49:17] STAT Training 1875/1875 - Loss 0.256 - LR 0.000273
[2020-05-14 16:49:49] STAT Validating 312/312 - Loss 0.292
[2020-05-14 16:49:49] STAT Epoch 13/20 - ETA 0:26:45 - loss/total=0.292 - ce=0.292 - ca=0.894
[2020-05-14 16:53:05] STAT Training 1875/1875 - Loss 0.240 - LR 0.000247
[2020-05-14 16:53:37] STAT Validating 312/312 - Loss 0.297
[2020-05-14 16:53:37] STAT Epoch 14/20 - ETA 0:22:55 - loss/total=0.297 - ce=0.297 - ca=0.897
[2020-05-14 16:56:54] STAT Training 1875/1875 - Loss 0.252 - LR 0.000223
[2020-05-14 16:57:26] STAT Validating 312/312 - Loss 0.296
[2020-05-14 16:57:26] STAT Epoch 15/20 - ETA 0:19:06 - loss/total=0.296 - ce=0.296 - ca=0.893
[2020-05-14 17:00:43] STAT Training 1875/1875 - Loss 0.248 - LR 0.000202
[2020-05-14 17:01:15] STAT Validating 312/312 - Loss 0.289
[2020-05-14 17:01:15] STAT Epoch 16/20 - ETA 0:15:16 - loss/total=0.289 - ce=0.289 - ca=0.899
[2020-05-14 17:04:30] STAT Training 1875/1875 - Loss 0.229 - LR 0.000183
[2020-05-14 17:05:02] STAT Validating 312/312 - Loss 0.291
[2020-05-14 17:05:02] STAT Epoch 17/20 - ETA 0:11:27 - loss/total=0.291 - ce=0.291 - ca=0.898
[2020-05-14 17:08:18] STAT Training 1875/1875 - Loss 0.232 - LR 0.000165
[2020-05-14 17:08:50] STAT Validating 312/312 - Loss 0.288
[2020-05-14 17:08:50] STAT Epoch 18/20 - ETA 0:07:38 - loss/total=0.288 - ce=0.288 - ca=0.898
[2020-05-14 17:12:06] STAT Training 1875/1875 - Loss 0.240 - LR 0.000150
[2020-05-14 17:12:38] STAT Validating 312/312 - Loss 0.291
[2020-05-14 17:12:38] STAT Epoch 19/20 - ETA 0:03:48 - loss/total=0.291 - ce=0.291 - ca=0.896
[2020-05-14 17:15:54] STAT Training 1875/1875 - Loss 0.220 - LR 0.000135
[2020-05-14 17:16:26] STAT Validating 312/312 - Loss 0.289
[2020-05-14 17:16:26] STAT Epoch 20/20 - ETA 0:00:00 - loss/total=0.289 - ce=0.289 - ca=0.898
Training done.

```

## After training

### Visual Verification

As we already have computed validation accuracy and cross entropy during training, we know that our model numerically does what we defined.
But you should always visually check if the predictions make sense.
It is possible, that your metric is wrongly implemented, or that your metric simply cannot capture all aspects.

Therefor, we simply run our visualisation we run before the training again.
Instead of balanced probabilities around 10% we expect one class to dominate all others at nearly 100%.

Example:
```python
# e.g. test/mnist_model.py
model.eval()
print("Training")
plt.figure(figsize=(16,6))
for i in range(12):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        pred = model.predict(**feat._asdict())
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        class_probs = softmax(pred.class_id)
        plt.title(", ".join(["{:.0f}".format(x*100) for x in class_probs]))
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()

print("Validation")
plt.figure(figsize=(16,6))
for i in range(12):
        idx = random.randint(0, len(train))
        feat, label = train[idx]
        pred = model.predict(**feat._asdict())
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        class_probs = softmax(pred.class_id)
        plt.title(", ".join(["{:.0f}".format(x*100) for x in class_probs]))
        plt.imshow(image_grid_unwrap(feat.features)[:,:,0], cmap='gray', vmin=0, vmax=255)
plt.show()
model.train()
```
Output:
```
Training
Validation

```
![data](../../docs/jlabdev_images/0728b0b4f9fd88e2b1f4bdc259499d4d.png)
![data](../../docs/jlabdev_images/26afabe5b6fb0ca1f2e9d208f5159398.png)

