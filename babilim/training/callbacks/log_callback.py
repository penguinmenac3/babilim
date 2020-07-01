import time

from babilim.core import ITensor
from babilim.core.logging import info, log_progress, status, create_checkpoint_structure, get_log_path, warn
from babilim.data import Dataloader
from babilim.core.module import Module
from babilim.training.callbacks.base_callback import BaseCallback
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer


def _dict_to_str(data):
    out = []
    for k in data:
        if isinstance(data[k], list):
            for i in data[k]:
                name = i.__name__
                out.append("{}_{}={:.3f}".format(k, name, data[k]))
        else:
            out.append("{}={:.3f}".format(k, data[k]))
    return " - ".join(out)


def _format_time(t):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


class LogCallback(BaseCallback):
    def __init__(self, train_log_steps=100):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None
        self.N = None
        self.train_log_steps = train_log_steps

    def on_fit_start(self, model: Module, train_dataloader: Dataloader, dev_dataloader: Dataloader, loss: Loss, optimizer: Optimizer, start_epoch: int, epochs: int) -> int:
        start_epoch = super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        info("Started fit.")
        self.start_time = time.time()
        log_progress(goal="warmup", progress=0, score=0)

        return start_epoch

    def on_fit_end(self) -> None:
        super().on_fit_end()

    def on_fit_interruted(self, exception) -> None:
        super().on_fit_interruted(exception)
        warn("Fit interrupted by user!")

    def on_fit_failed(self, exception) -> None:
        super().on_fit_failed(exception)

    def on_epoch_begin(self, dataloader: Dataloader, phase: str, epoch: int) -> None:
        super().on_epoch_begin(dataloader, phase, epoch)
        self.N = len(dataloader)
        self.epoch_start_time = time.time()
        log_progress(goal=phase, progress=0, score=0)

    def on_iter_begin(self, iter: int, feature, target) -> None:
        super().on_iter_begin(iter, feature, target)

    def on_iter_end(self, predictions, loss_result: ITensor) -> None:
        elapsed_time = time.time() - self.epoch_start_time
        eta = elapsed_time / (self.iter + 1) * (self.N - (self.iter + 1))
        if self.phase == "train":
            header = "Training"
        else:
            header = "Dev Evaluation"
        status("{} {}/{} (ETA {}) - Loss {:.3f} - LR {:.6f}".format(header, self.iter + 1, self.N, _format_time(eta), self.loss.avg["loss/total"], self.optimizer.lr), end="")
        if self.iter % self.train_log_steps == self.train_log_steps - 1:
            log_progress(goal="{} {}/{}".format(self.phase, self.epoch, self.epochs), progress=(self.iter + 1) / self.N, score=self.loss.avg["loss/total"])
        super().on_iter_end(predictions, loss_result)

    def on_epoch_end(self) -> None:
        if self.phase != "train":
            elapsed_time = time.time() - self.start_time
            eta = elapsed_time / (self.epoch + 1) * (self.epochs - (self.epoch + 1))
            status("Epoch {}/{} (ETA {}) - {}".format(self.epoch + 1, self.epochs, _format_time(eta), _dict_to_str(self.loss.avg)))
        else:
            status("Training {}/{} - {} - LR {:.6f}".format(self.epoch + 1, self.epochs, _dict_to_str(self.loss.avg), self.optimizer.lr))

        log_progress(goal="{} {}/{}".format(self.phase, self.epoch, self.epochs), progress=1, score=self.loss.avg["loss/total"])
        super().on_epoch_end()
