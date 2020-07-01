from babilim.core import ITensor
from babilim.data import Dataloader
from babilim.core.module import Module
from babilim.training.callbacks.base_callback import BaseCallback
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer, LearningRateSchedule


class LearningRateUpdateCallback(BaseCallback):
    def __init__(self, learning_rate_schedule: LearningRateSchedule, train_log_steps=100, initial_samples_seen=0):
        super().__init__()
        self.learning_rate_schedule = learning_rate_schedule
        self.train_log_steps = train_log_steps
        self.samples_seen = initial_samples_seen

    def on_fit_start(self, model: Module, train_dataloader: Dataloader, dev_dataloader: Dataloader, loss: Loss, optimizer: Optimizer, start_epoch: int, epochs: int) -> int:
        start_epoch = super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
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
        self.optimizer.lr = self.learning_rate_schedule(self.samples_seen / self.feature[0].shape[0])

    def on_iter_end(self, predictions, loss_result: ITensor) -> None:
        self.samples_seen += self.feature[0].shape[0]
        super().on_iter_end(predictions, loss_result)

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
