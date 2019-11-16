from typing import Any
import numpy as np
from collections import defaultdict
from babilim.core import ITensor, Tensor, StatefullObject


class Metrics(StatefullObject):
    def __init__(self, log_std=False, log_min=False, log_max=False):
        super().__init__()
        self._accumulators = defaultdict(list)
        self._log_std = log_std
        self._log_min = log_min
        self._log_max = log_max

    def __call__(self,
                 y_pred: Any,
                 y_true: Any) -> None:
        """
        Implement a couple of metrics function between preds and true outputs.

        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        self.call(y_pred, y_true)

    def call(self,
                 y_pred: Any,
                 y_true: Any) -> None:
        """
        Implement a couple of metrics function between preds and true outputs.

        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        raise NotImplementedError("Every metric must implement the call method.")

    def log(self, name: str, value: ITensor):
        val = value.numpy()
        if len(val.shape) > 0:
            self._accumulators[name].append(val)
        else:
            self._accumulators[name].append(np.array([val]))

    def reset_avg(self):
        self._accumulators = defaultdict(list)

    def summary(self, samples_seen, summary_writer=None):
        if summary_writer is not None:
            for k in self._accumulators:
                combined = np.concatenate(self._accumulators[k], axis=0)
                summary_writer.add_scalar("{}".format(k), combined.mean(), global_step=samples_seen)
                if self._log_std:
                    summary_writer.add_scalar("{}_std".format(k), combined.std(), global_step=samples_seen)
                if self._log_min:
                    summary_writer.add_scalar("{}_min".format(k), combined.min(), global_step=samples_seen)
                if self._log_max:
                    summary_writer.add_scalar("{}_max".format(k), combined.max(), global_step=samples_seen)
        else:
            import tensorflow as tf
            for k in self._accumulators:
                combined = np.concatenate(self._accumulators[k], axis=0)
                tf.summary.scalar("{}".format(k), combined.mean(), step=samples_seen)
                if self._log_std:
                    tf.summary.scalar("{}_std".format(k), combined.std(), step=samples_seen)
                if self._log_min:
                    tf.summary.scalar("{}_min".format(k), combined.min(), step=samples_seen)
                if self._log_max:
                    tf.summary.scalar("{}_max".format(k), combined.max(), step=samples_seen)

    @property
    def avg(self):
        avgs = {}
        for k in self._accumulators:
            combined = np.concatenate(self._accumulators[k], axis=0)
            avgs[k] = combined.mean()
        return avgs


class NativeMetricsWrapper(Metrics):
    def __init__(self, metrics, log_std=False, log_min=False, log_max=False):
        """
        Wrap a native metrics as a babilim metrics.

        The wrapped object must have the following signature:

            Callable(y_pred, y_true, log_val) -> None

        where log_val will be a function which can be used for logging scalar tensors/values.

        :param metrics: The metrics that should be wrapped.
        """
        super().__init__(log_std=log_std, log_min=log_min, log_max=log_max)
        self.metrics = metrics

    def call(self, y_pred: Any, y_true: Any) -> None:
        # Unwrap arguments
        tmp = y_true._asdict()
        y_true_tmp = {k: tmp[k].native for k in tmp}
        y_true = type(y_true)(**y_true_tmp)

        tmp = y_pred._asdict()
        y_pred_tmp = {k: tmp[k].native for k in tmp}
        y_pred = type(y_pred)(**y_pred_tmp)

        # call function
        self.metrics(y_pred=y_pred, y_true=y_true,
                     log_val=lambda name, tensor: self.log(name, Tensor(data=tensor, trainable=True)))
