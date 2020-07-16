import os

import babilim
from babilim.core.checkpoint import save_state, load_state
from babilim.core.logging import get_log_path, info, warn, create_checkpoint_structure
from babilim.data import Dataloader
from babilim.core.module import Module
from babilim.training.callbacks.base_callback import BaseCallback
from babilim.training.losses import Loss
from babilim.training.optimizers import Optimizer


class CheckpointCallback(BaseCallback):
    def __init__(self, keep_only_best_and_latest=True, native_format=False):
        super().__init__()
        self.keep_only_best_and_latest = keep_only_best_and_latest
        self.native_format = native_format
        self.best_loss = None

    def on_fit_start(self, model: Module, train_dataloader: Dataloader, dev_dataloader: Dataloader, loss: Loss, optimizer: Optimizer, start_epoch: int, epochs: int) -> int:
        super().on_fit_start(model, train_dataloader, dev_dataloader, loss, optimizer, start_epoch, epochs)
        if get_log_path() is None:
            raise RuntimeError("You must setup logger before calling the fit method. See babilim.core.logging.set_logger")
        create_checkpoint_structure()

        # Actually force model to be build by running one forward step
        if not getattr(model, "initialized_model", False):
            if babilim.core.logging.DEBUG_VERBOSITY:
                info("Build Model")
            model.initialized_model = True
            features, _ = next(iter(train_dataloader))
            model(**features._asdict())

        # Load Checkpoint
        saved_models_path = os.path.join(get_log_path(), "checkpoints")
        saved_models = sorted(
            [os.path.join(saved_models_path, f) for f in os.listdir(saved_models_path) if not f.startswith("best")])
        if len(saved_models) > 0 and os.path.exists(saved_models[-1]):
            info("Loading checkpoint: {}".format(saved_models[-1]))
            checkpoint = load_state(saved_models[-1], native_format=self.native_format)
            if babilim.core.logging.DEBUG_VERBOSITY:
                checkpoint.print()
            start_epoch = checkpoint["epoch"] + 1
            if "model" in checkpoint:
                if babilim.core.logging.DEBUG_VERBOSITY:
                    info("Load Model...")
                model.load_state_dict(checkpoint["model"])
            else:
                warn("Could not find model_state in checkpoint.")
            if "optimizer" in checkpoint:
                if babilim.core.logging.DEBUG_VERBOSITY:
                    info("Load Optimizer...")
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                warn("Could not find optimizer_state in checkpoint.")
            if "loss" in checkpoint:
                if babilim.core.logging.DEBUG_VERBOSITY:
                    info("Load Loss...")
                self.loss.load_state_dict(checkpoint["loss"])
            else:
                warn("Could not find loss_state in checkpoint.")

        if babilim.core.logging.DEBUG_VERBOSITY:
            info("Trainable Variables:")
            for name, var in model.named_trainable_variables.items():
                info("  {}: {}".format(name, var.shape))
            for name, var in self.loss.named_trainable_variables.items():
                info("  {}: {}".format(name, var.shape))
            info("Untrainable Variables:")
            for name, var in model.named_untrainable_variables.items():
                info("  {}: {}".format(name, var.shape))
            for name, var in self.loss.named_untrainable_variables.items():
                info("  {}: {}".format(name, var.shape))

        return start_epoch

    def on_epoch_end(self) -> None:
        if self.keep_only_best_and_latest:
            if self.phase == "train":
                self.save("latest")
            else:
                current_loss = self.loss.avg["loss/total"]
                if self.best_loss is None or current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save("best")
        else:
            if self.phase == "train":
                self.save("epoch_{:09d}".format(self.epoch))

        super().on_epoch_end()

    def save(self, name: str) -> None:
        chkpt_extension = ".npz"
        if self.native_format:
            if babilim.is_backend(babilim.TF2_BACKEND):
                chkpt_extension = ""
            elif babilim.is_backend(babilim.PYTORCH_BACKEND):
                chkpt_extension = ".pth"
        checkpoint_sub_path = os.path.join("checkpoints", "{}{}".format(name, chkpt_extension))
        checkpoint_path = os.path.join(get_log_path(), checkpoint_sub_path)
        save_state({
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": self.loss.state_dict()
        }, checkpoint_path, native_format=self.native_format)
        info("Saved Checkoint: {}".format(checkpoint_sub_path))
