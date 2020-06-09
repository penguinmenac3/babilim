import os

from tensorboardX import SummaryWriter

from babilim.core import ITensor
from babilim.core.logging import get_log_path, create_checkpoint_structure
from babilim.data import Dataloader
from babilim.model.module import Module
from babilim.training.callbacks.base_callback import BaseCallback
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer


class TensorboardCallback(BaseCallback):
    def __init__(self, train_log_steps=100, initial_samples_seen=0):
        super().__init__()
        self.train_log_steps = train_log_steps
        self.samples_seen = initial_samples_seen
        self.train_summary_writer = None
        self.dev_summary_writer = None

    def on_fit_start(self, model: Module, train_dataloader: Dataloader, dev_dataloader: Dataloader, loss: Loss, optimizer: Optimizer, start_epoch: int, epochs: int) -> int:
        start_epoch = super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        self.train_summary_writer = SummaryWriter(os.path.join(get_log_path(), "train"))
        self.dev_summary_writer = SummaryWriter(os.path.join(get_log_path(), "dev"))
        return start_epoch

    def on_fit_end(self) -> None:
        super().on_fit_end()

    def on_fit_interruted(self, exception) -> None:
        super().on_fit_interruted(exception)

    def on_fit_failed(self, exception) -> None:
        super().on_fit_failed(exception)

    def on_epoch_begin(self, dataloader: Dataloader, phase: str, epoch: int) -> None:
        super().on_epoch_begin(dataloader, phase, epoch)

    def on_iter_begin(self, iter: int, feature, target) -> None:
        super().on_iter_begin(iter, feature, target)

    def on_iter_end(self, predictions, loss_result: ITensor) -> None:
        self.samples_seen += self.feature[0].shape[0]
        if self.phase == "train" and self.iter % self.train_log_steps == self.train_log_steps - 1:
            self.loss.summary(self.samples_seen, self.train_summary_writer)
            self.loss.reset_avg()
        super().on_iter_end(predictions, loss_result)

    def on_epoch_end(self) -> None:
        if self.phase == "train":
            self.loss.summary(self.samples_seen, self.train_summary_writer)
        else:
            self.loss.summary(self.samples_seen, self.dev_summary_writer)
        self.loss.reset_avg()
        super().on_epoch_end()
