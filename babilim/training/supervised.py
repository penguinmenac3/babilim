from typing import Iterable, Union

from babilim.training.trainer import Trainer

from babilim.data import Dataloader
from babilim.core import GradientTape
from babilim.core.logging import error
from babilim.model.module import Module
from babilim.training.callbacks.checkpoint_callback import CheckpointCallback
from babilim.training.callbacks.log_callback import LogCallback
from babilim.training.callbacks.tensorboard_callback import TensorboardCallback
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer
from babilim.training.callbacks.base_callback import BaseCallback

DEFAULT_CALLBACKS = [
    LogCallback(),
    CheckpointCallback(),
    TensorboardCallback()
]


class SupervisedTrainer(Trainer):
    def __init__(self, model: Module, loss: Loss, optimizer: Optimizer, callbacks: Iterable[BaseCallback] = DEFAULT_CALLBACKS):
        """
        Create a trainer for supervised training scenarios.

        The fit function is very basic and can be vastly extended by using callbacks.
        The default behaviour can be changed by changing not passing the DEFAULT_CALLBACKS but a modified set of callbacks (only do this if you know what you are doing).
        A normal use case would be to simply add some callbacks:
            SupervisedTrainer(callbacks=DEFAULT_CALLBACKS + [my_callback])

        :param model: The model that should be fit.
        :param loss: The loss defines a what should optimization.
        :param optimizer: The optimizer defines how the optimization is done.
        :param callbacks: Any callbacks that you want to add. You should always write callbacks=DEFAULT_CALLBACKS+[MyCallback], otherwise the default callbacks will not be called.
        Callbacks will be called in the order as specified in this list. So make sure your callbacks are in the correct order (and when in doubt DEFAULT_CALLBACKS first, yours later).
        """
        self.callbacks = callbacks
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def run_epoch(self, dataloader: Dataloader, phase: str, epoch: int):
        """
        Run an epoch in training or validation.

        (This function is called in fit and it is NOT RECOMMENDED to use this function from outside.)

        Optimizer is "optional" if it is set to None, it is a validation run otherwise it is a training run.

        :param dataloader: The dataloader created from a dataset.
        :param phase: The phase (train/dev/test) which is used for running.
        :param epoch: The epoch number.
        :return: Returns the average loss.
        """
        if self.model is None:
            raise RuntimeError("You must compile the trainer first!")
        self.loss.reset_avg()
        for callback in self.callbacks:
            callback.on_epoch_begin(dataloader, phase, epoch)

        # Setup the training loop
        variables = self.model.trainable_variables + self.loss.trainable_variables

        # Loop over the dataset_class and update weights.
        for iter, (x, y) in enumerate(dataloader):
            for callback in self.callbacks:
                callback.on_iter_begin(iter, x, y)

            # Forward pass, computing gradients and applying them
            with GradientTape(variables) as tape:
                predictions = self.model(**x._asdict())
                for name, p in predictions._asdict().items():
                    if p.is_nan().any():
                        error("NaN NetworkOutput {}: {}".format(name, p.native))
                        raise ValueError("NetworkOutput {} got nan.".format(name))
                loss_result = self.loss(y_true=y, y_pred=predictions)
                self.loss.log("loss/total", loss_result)
                if loss_result.is_nan().any():
                    error("NaN Loss")
                    raise ValueError("Loss got nan.")
            gradients = tape.gradient(loss_result)

            if phase == "train":
                self.optimizer.apply_gradients(gradients, variables)

            for callback in self.callbacks:
                callback.on_iter_end(predictions, loss_result)

        for callback in self.callbacks:
            callback.on_epoch_end()

    def fit(self, train_dataloader: Dataloader, dev_dataloader: Dataloader, epochs: int):
        """
        Fit the model managed by this trainer to the data.

        :param train_dataloader: The dataloader for training your neural network (train split).
        :param dev_dataloader: The dataloader for validation during your development (dev split). NOT TEST SPLIT!
        :param epochs: The number of epochs describes how often the fit will iterate over the dataloaders.
        """
        try:
            start_epoch = 0
            for callback in self.callbacks:
                start_epoch = callback.on_fit_start(self.model, train_dataloader, dev_dataloader, self.loss, self.optimizer, start_epoch, epochs)

            for epoch in range(start_epoch, epochs):
                self.model.train()
                self.run_epoch(train_dataloader, "train", epoch)
                self.model.eval()
                self.run_epoch(dev_dataloader, "dev", epoch)
        except KeyboardInterrupt as e:
            for callback in self.callbacks:
                callback.on_fit_interruted(e)
        except Exception as e:
            for callback in self.callbacks:
                callback.on_fit_failed(e)
            raise e

        for callback in self.callbacks:
            callback.on_fit_end()
