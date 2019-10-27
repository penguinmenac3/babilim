from typing import Any
import numpy as np
import babilim
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor
from babilim.core.statefull_object import StatefullObject


class Loss(StatefullObject):
    def __init__(self):
        super().__init__("Loss")
        self._accumulators = {}
        self._counters = {}

    def __call__(self,
                 y_pred: Any,
                 y_true: Any) -> ITensor:
        """
        Implement a loss function between preds and true outputs.

        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        loss = self.call(y_pred, y_true)
        if loss.is_nan().any():
            raise ValueError("Loss '{}' is nan. Loss value: {}".format(self.name, loss))
        return loss

    def call(self,
                 y_pred: Any,
                 y_true: Any) -> ITensor:
        """
        Implement a loss function between preds and true outputs.

        :param y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        :param y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
        """
        raise NotImplementedError("Every loss must implement the call method.")

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
                summary_writer.add_scalar("{}".format(k), avgs[k].numpy(), global_step=samples_seen)
        else:
            import tensorflow as tf
            for k in avgs:
                tf.summary.scalar("{}".format(k), avgs[k].numpy(), step=samples_seen)

    @property
    def avg(self):
        avgs = {}
        for k in self._accumulators:
            avgs[k] = self._accumulators[k] / self._counters[k]
        return avgs


class NativeLossWrapper(Loss):
    def __init__(self, loss):
        """
        Wrap a native loss as a babilim loss.

        The wrapped object must have the following signature:

            Callable(y_pred, y_true, log_val) -> Tensor

        where log_val will be a function which can be used for logging scalar tensors/values.

        :param loss: The loss that should be wrapped.
        """
        super().__init__()
        self.loss = loss

    def call(self, y_pred: Any, y_true: Any) -> ITensor:
        # Unwrap arguments
        tmp = y_true._asdict()
        y_true_tmp = {k: tmp[k].native for k in tmp}
        y_true = type(y_true)(**y_true_tmp)

        tmp = y_pred._asdict()
        y_pred_tmp = {k: tmp[k].native for k in tmp}
        y_pred = type(y_pred)(**y_pred_tmp)

        # call function
        result = self.loss(y_pred=y_pred, y_true=y_true,
                           log_val=lambda name, tensor: self.log(name, Tensor(data=tensor, trainable=True)))

        return Tensor(data=result, trainable=True)


class SparseCrossEntropyLossFromLogits(Loss):
    def __init__(self):
        super().__init__()
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            from torch.nn import CrossEntropyLoss
            self.loss_fun = CrossEntropyLoss()
        else:
            from tensorflow.nn import sparse_softmax_cross_entropy_with_logits
            self.loss_fun = sparse_softmax_cross_entropy_with_logits

    def call(self, y_pred: ITensor, y_true: ITensor) -> ITensor:
        y_true = y_true.cast("int64")
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            return Tensor(data=self.loss_fun(y_pred.native, y_true.native), trainable=True)
        else:
            return Tensor(data=self.loss_fun(labels=y_true.native, logits=y_pred.native), trainable=True)


class MeanSquaredError(Loss):
    def call(self, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor:
        return ((y_pred - y_true) ** 2).mean(axis=axis)


class SparseCategoricalAccuracy(Loss):
    def call(self, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor:
        pred_class = y_pred.argmax(axis=axis)
        true_class = y_true.cast("int64")
        correct_predictions = pred_class == true_class
        return correct_predictions.cast("float32").mean()
