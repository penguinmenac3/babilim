from typing import Any
import numpy as np
from babilim.core import ITensor, Tensor, StatefullObject


class Metrics(StatefullObject):
    def __init__(self):
        super().__init__("Metrics")
        self._accumulators = {}
        self._counters = {}

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
        if name not in self._accumulators:
            self._accumulators[name] = value.copy()
            self._counters[name] = 1
        else:
            self._accumulators[name] += value.stop_gradients()
            self._counters[name] += 1

    def reset_avg(self):
        for k in self._accumulators:
            acc = self._accumulators[k]
            acc.assign(np.zeros_like(acc.numpy()))
            self._counters[k] = 0

    def summary(self, samples_seen, summary_writer=None):
        avgs = self.avg
        if summary_writer is not None:
            for k in avgs:
                summary_writer.add_scalar("{}_metric".format(k), avgs[k].numpy(), global_step=samples_seen)
        else:
            import tensorflow as tf
            for k in avgs:
                tf.summary.scalar("{}_metric".format(k), avgs[k].numpy(), step=samples_seen)

    @property
    def avg(self):
        avgs = {}
        for k in self._accumulators:
            avgs[k] = self._accumulators[k] / self._counters[k]
        return avgs


class NativeMetricsWrapper(Metrics):
    def __init__(self, metrics):
        """
        Wrap a native metrics as a babilim metrics.

        The wrapped object must have the following signature:

            Callable(y_pred, y_true, log_val) -> None

        where log_val will be a function which can be used for logging scalar tensors/values.

        :param metrics: The metrics that should be wrapped.
        """
        super().__init__()
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
