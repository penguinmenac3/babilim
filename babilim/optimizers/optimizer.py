from typing import Iterable
from babilim.core.itensor import ITensor
from babilim.core.statefull_object import StatefullObject


class Optimizer(StatefullObject):
    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor], learning_rate: float) -> None:
        """
        This method applies the gradients to variables.

        :param gradients: An interable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        :param learning_rate: The learning rate which is currently used.
        """
        raise NotImplementedError("Apply gradients must be implemented by every optimizer.")
