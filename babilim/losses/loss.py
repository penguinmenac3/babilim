from typing import Any
import numpy as np
import babilim
from babilim.core.itensor import ITensor
from babilim.core.tensor import Tensor


class Loss(object):
    def __init__(self):
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



class CrossEntropyLossFromLogits(Loss):
    def __init__(self):
        super().__init__()
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            from torch.nn import CrossEntropyLoss
            # FIXME This is actually the wrong one
            self.loss_fun = CrossEntropyLoss()
        else:
            from tensorflow.nn import softmax_cross_entropy_with_logits
            self.loss_fun = softmax_cross_entropy_with_logits

    def call(self, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor:
        if babilim.is_backend(babilim.PYTORCH_BACKEND):
            return Tensor(data=None, trainable=True, native=self.loss_fun(y_pred.native, y_true.native))
        else:
            return Tensor(data=None, trainable=True, native=self.loss_fun(labels=y_true.native, logits=y_pred.native, axis=axis))


class MeanSquaredError(Loss):
    def call(self, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor:
        return ((y_pred - y_true) ** 2).mean(axis=axis)


class CategoricalAccuracy(Loss):
    def call(self, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor:
        pred_class = y_pred.argmax(axis=axis)
        true_class = y_true.argmax(axis=axis)
        correct_predictions = pred_class == true_class
        return correct_predictions.cast("float32").mean()
