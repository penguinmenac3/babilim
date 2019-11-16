import os
import numpy as np
from babilim import warn, info


class Checkpoint(object):
    def __init__(self, checkpoint_path):
        self.checkoint_path = checkpoint_path
        self.data = {}
        if os.path.exists(checkpoint_path):
            self.data = np.load(checkpoint_path, allow_pickle=False)
            try:
                if "model_state_dict" in self.data:
                    raise ValueError
            except ValueError:
                warn("Checkpoint format deprecated. Save this checkpoint to update the format.")
                self.data = {}
                checkpoint = np.load(checkpoint_path, allow_pickle=True)
                if "epoch" in checkpoint:
                    self.set_epoch(checkpoint["epoch"][()])
                if "model_state_dict" in checkpoint:
                    self.set_state_dict("model", checkpoint["model_state_dict"][()])
                if "optimizer_state_dict" in checkpoint:
                    self.set_state_dict("optimizer", checkpoint["optimizer_state_dict"][()])
                if "loss_state_dict" in checkpoint:
                    self.set_state_dict("loss", checkpoint["loss_state_dict"][()])
                if "metrics_state_dict" in checkpoint:
                    self.set_state_dict("metrics", checkpoint["metrics_state_dict"][()])
                if "lr_schedule_state_dict" in checkpoint:
                    self.set_state_dict("lr_schedule", checkpoint["lr_schedule_state_dict"][()])

    def print(self):
        info("Checkpoint: {}".format(self.checkoint_path))
        for k in self.data:
            if isinstance(self.data[k], np.ndarray):
                info("  {}: {}".format(k, self.data[k].shape))
            else:
                info("  {}: {}".format(k, self.data[k]))

    def save(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkoint_path
        np.savez_compressed(checkpoint_path, **self.data)

    def set_epoch(self, epoch):
        self.data["epoch"] = epoch

    def get_epoch(self):
        return self.data["epoch"]

    def get_state_dict(self, prefix):
        tmp = self.data
        tmp = {"{}".format("/".join(k.split("/")[1:])): tmp[k] for k in tmp if k.startswith(prefix)}
        return tmp

    def set_state_dict(self, prefix, state_dict):
        tmp = {"{}/{}".format(prefix, k): state_dict[k] for k in state_dict}
        self.data.update(tmp)
