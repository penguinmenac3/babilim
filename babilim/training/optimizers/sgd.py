# AUTOGENERATED FROM: babilim/training/optimizers/sgd.ipynb

# Cell: 0
"""
# babilim.training.optimizers.sgd

> Common stochastic gradient descent optimizer.
"""

# Cell: 1
from typing import Iterable
from babilim.core.itensor import ITensor
from babilim.training.optimizers.optimizer import Optimizer


# Cell: 2
class SGD(Optimizer):
    def __init__(self, initial_lr: float, momentum: float=0.95, dampening: float=0.0, weight_decay: float=0, nesterov: bool=True):
        """
        Common stochastic gradient descent optimizer.
        
        :param initial_lr: The initial learning rate for the optimizer. Learning rates are updated in the optimizer via callbacks.
        :param momentum: Value between 0 and 1 representing the momentum of the old grads to keep.
        :param dampening: Value between 0 and 1 representing by how much the accumulated gradients should be dampened.
        :param weight_decay: Value between 0 and 1 representing by how much the new gradients should be reduced.
        :param nesterov: If nesterov momentum should be used.
        """
        super().__init__(initial_lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.dampening = dampening
        self.state = {}

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        """
        This method applies the gradients to variables.

        :param gradients: An iterable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        :param learning_rate: The learning rate which is currently used.
        """
        for grad, var in zip(gradients, variables):
            if grad is None:
                continue
            d_p = grad
            if self.weight_decay != 0:
                d_p += var * self.weight_decay
            if self.momentum != 0:
                if var.ref() not in self.state:
                    buf = self.state[var.ref()] = d_p.copy()
                else:
                    buf = self.state[var.ref()]
                    buf *= self.momentum
                    buf += d_p * (1 - self.dampening)
                if self.nesterov:
                    d_p += buf * self.momentum
                else:
                    d_p = buf

            var -= d_p * self.lr
