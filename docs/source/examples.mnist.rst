MNIST Example (Babilim)
=======================

This example shows how to implement a simple network solving MNIST with babilim.
The focus is not to create the shortest possible solution, but rather to show how babilim works.
The example is build in a way it can be applied to other problems as well.

The full code can be found `here on github <https://github.com/penguinmenac3/babilim/blob/master/examples/fashion_mnist.py>`_.

**1. Define the Problem to solve**

Even before you start coding anything, you should start to define the problem you want to solve on a piece of paper.
This will help your code quality a lot.

In this tutorial we will solve FashionMnist, classification of 28x28 Pixel fashion images.

**2. Define your Inputs and Outputs**

With a goal established we can define inputs and outputs to our network.
For that we create a named tuple for input and output, with features and class_id respectively.

.. code-block:: python

    from collections import namedtuple

    # Create some named tuple for our inputs and outputs so we do not confuse them.
    NetworkInput = namedtuple("NetworkInput", ["features"])
    NetworkOutput = namedtuple("NetworkOutput", ["class_id"])

**3. Create a configuration**

To further define our problem we create a configuration to use for our problem and training.
This configuration is something you should create at the start and change and extend as you go.
For the minimal necessary configuration look at the "Experiment Configuration" section of this documentation.

Since we are solving MNIST we setup our number of categories to 10 (for later use in our network) and define a place where to save our dataset.
Furthermore all training parameters that are required get defined.

.. code-block:: python

    import babilim.optimizers.learning_rates as lr
    from babilim.experiment import Config

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

**4. Write your dataset loader**

Next step is writing your dataset loader.
You should take your time with this step and thoroughly test your implementation, visually and with test cases.
In this tutorial we will not test but just provide an example on how to write one.

Since this tutorial should work for both pytorch and tensorflow we will write a wrapper around the FashionMNIST dataset provided by the frameworks.
This wrapper will show the concept of how to write platform specific parts of the code.

In the __init__ function we start by calling the super. And then handling the tensorflow case.
For that we check if the backend is tensorflow and then load fashion mnist from tensorflow.
Note the import being only in the tensorflow case because it would break for a pytorch environment.
After saving the variables as local variables we implement the pytorch case, where we simply import the pytorch mnist dataset implementation and use it.
The problem_base_dir from the configuration is used to store the dataset.

In the __len__ function we return the number of training samples our dataset has, dependant on it being for training or validation.

The getitem function actually returns the training sample.
Here we read the example from the stored array.
More complex datasets might require loading from the disk here.
Then, we transform the data to have the shape we want, before the image data is wrapped using image_grid_wrap and then used in our named tuple we defined in step 1.
Your dataset should always output two namedtuples one for the features and one for the labels.

At the end we define a version of our dataset.
This is important for larger projects where your dataset might generate caches which you need to check for validity.


.. code-block:: python

    import babilim
    from babilim import TF_BACKEND, PYTORCH_BACKEND
    from babilim.data import Dataset

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

**5. Defining the model**

Finally, after we have our dataset working and tested, we can define our model.

A model has three parts to it:

* the initializer __init__ where the configuration is stored as part of the object,
* the build function which creates the layers
* and the call function which contains the forward pass of the network.

Here we can see that the init is pretty empty apart from creating the python variables required.
The build function creates all the layers and appends them to our linear list of layers.
In the call function all layers in linear are executed one after another in a for loop.

The inputs of the build and call are the fields of NetworkInputs defined in step 1.
The return value of the call function is of type NetworkOutput.
This makes your network type safe and avoids some unwanted surprises.

Creation of layers works like in keras as it is the simplest way.
Not that in contrast to native pytorch you do not need to define the input shapes of a layer.

.. code-block:: python

    from babilim.core import ITensor, RunOnlyOnce
    from babilim.layers import BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Linear, ReLU, Flatten
    from babilim.models import IModel

    class FashionMnistModel(IModel):
        def __init__(self, config: FashionMnistConfig, name: str = "FashionMnistModel"):
            super().__init__(name, layer_type="FashionMnistModel")
            self.config = config
            self.linear = []

        @RunOnlyOnce
        def build(self, features: ITensor):
            out_features = self.config.problem_number_of_categories

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

        def call(self, features: ITensor) -> NetworkOutput:
            net = features
            for l in self.linear:
                net = l(net)
            return NetworkOutput(class_id=net)

**6. Defining the Loss and Metrics**

With a model, the last step before training is to setup some losses and metrics.

The loss is pretty simple. It is a class implementing a call function which has two parameters.
The first parameter is y_pred representing the actual network output, and y_true is the intended network output as returned by the dataset.
Intermediate computations such as partial losses can be logged using self.log(name, tensor).
The return type of the loss is a single Tensor, the loss that should be optimized.

.. code-block:: python

    from babilim.core import ITensor
    from babilim.losses import Loss, SparseCrossEntropyLossFromLogits

    class FashionMnistLoss(Loss):
        def __init__(self):
            super().__init__()
            self.ce = SparseCrossEntropyLossFromLogits()

        def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> ITensor:
            return self.ce(y_pred.class_id, y_true.class_id).mean()

The metric is very similar to the loss. It is a class implementing a call function which has two parameters.
The first parameter is y_pred representing the actual network output, and y_true is the intended network output as returned by the dataset.
However, it does not have a return type and no effect on the optimization.
Whereas it has no return type values must be explicitly logged using self.log(name, tensor).

.. code-block:: python

    from babilim.losses import Metrics, SparseCrossEntropyLossFromLogits, SparseCategoricalAccuracy

    class FashionMnistMetrics(Metrics):
        def __init__(self):
            super().__init__()
            self.ce = SparseCrossEntropyLossFromLogits()
            self.ca = SparseCategoricalAccuracy()

        def call(self, y_pred: NetworkOutput, y_true: NetworkOutput) -> None:
            self.log("ce", self.ce(y_pred.class_id, y_true.class_id).mean())
            self.log("ca", self.ca(y_pred.class_id, y_true.class_id).mean())


**7. Training it**

Finally we can write code which glues everything together.
First select your backend of choice.
Then, create a configuration and use it to setup the logger module.

After that you can create your dataset for training and validation by instantiating the class created in step 4.
Also our model from step 5 can be instantiated as well as the loss and metrics from step 6.
Finally we select an optimizer (typically SGD is fine).

With all objects instantiated, we can call the fit method on the model to actually fit the model against the data using our configuration, optimizer and learning rate schedule.

.. code-block:: python

    import babilim
    import babilim.logger as logger
    from babilim.optimizers import SGD
    from babilim import PYTORCH_BACKEND, TF_BACKEND, PHASE_TRAIN, PHASE_VALIDATION

    babilim.set_backend(PYTORCH_BACKEND)

    # Create our configuration (containing all hyperparameters)
    config = FashionMnistConfig()
    logger.setup(config, continue_training=False)

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

**8. What next?**

Solve your own problem in a similar manner.
Dive into the detailed api documentation and even have peeks at the code to become a true master in using babilim.
