from babilim.core import ITensor
from babilim.data import Dataloader
from babilim.model.module import Module
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer


class BaseCallback(object):
    def __init__(self):
        self.model = None
        self.train_dataloader = None
        self.dev_dataloader = None
        self.loss = None
        self.optimizer = None
        self.epochs = None
        self.phase = None
        self.epoch = None
        self.active_dataloader = None
        self.iter = None
        self.feature = None
        self.target = None

    def on_fit_start(self, model: Module, train_dataloader: Dataloader, dev_dataloader: Dataloader, loss: Loss, optimizer: Optimizer, start_epoch: int, epochs: int) -> int:
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        return start_epoch

    def on_fit_end(self) -> None:
        self.model = None
        self.train_dataloader = None
        self.dev_dataloader = None
        self.loss = None
        self.optimizer = None
        self.epochs = None

    def on_fit_interruted(self, exception) -> None:
        return

    def on_fit_failed(self, exception) -> None:
        return

    def on_epoch_begin(self, dataloader: Dataloader, phase: str, epoch: int) -> None:
        self.active_dataloader = dataloader
        self.phase = phase
        self.epoch = epoch

    def on_iter_begin(self, iter: int, feature, target) -> None:
        self.iter = iter
        self.feature = feature
        self.target = target

    def on_iter_end(self, predictions, loss_result: ITensor) -> None:
        self.iter = None
        self.feature = None
        self.target = None

    def on_epoch_end(self) -> None:
        self.active_dataloader = None
        self.phase = None
        self.epoch = None
