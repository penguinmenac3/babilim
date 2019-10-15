from typing import Iterable
from babilim.core.itensor import ITensor
from babilim.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, momentum: float=0.95, dampening: float=0.0, weight_decay: float=0, nesterov: bool=True):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.state = {}

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor], learning_rate: float) -> None:
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            d_p = grad
            if self.weight_decay != 0:
                d_p += var * self.weight_decay
            if self.momentum != 0:
                if var.name not in self.state:
                    buf = self.state[var.name] = d_p.copy()
                else:
                    buf = self.state[var.name]
                    buf *= self.momentum
                    buf += d_p * (1 - self.dampening)
                if self.nesterov:
                    d_p += buf * self.momentum
                else:
                    d_p = buf

            var -= d_p * learning_rate
