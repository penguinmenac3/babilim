from typing import Any
import numpy as np
from babilim.core.itensor import ITensor


class Metrics(object):
    def __init__(self):
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
            acc.assign(np.zeroes_like(acc.numpy()))
            self._counters[k] = 0

    def summary(self):
        avgs = self.avg
        print("Metrics: ", end="")
        for k in avgs:
            print("{}={:.4f} ".format(k, avgs[k].numpy()), end="")
        print()

    @property
    def avg(self):
        avgs = {}
        for k in self._accumulators:
            avgs[k] = self._accumulators[k] / self._counters[k]
        return avgs
