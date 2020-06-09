from typing import Iterable
from babilim.core.itensor import ITensor
from babilim.core.statefull_object import StatefullObject
from babilim.core import RunOnlyOnce


class Optimizer(StatefullObject):
    def __init__(self, initial_lr: float):
        super().__init__()
        self.lr = initial_lr

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        """
        This method applies the gradients to variables.

        :param gradients: An interable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        :param learning_rate: The learning rate which is currently used.
        """
        raise NotImplementedError("Apply gradients must be implemented by every optimizer.")


class NativePytorchOptimizerWrapper(Optimizer):
    def __init__(self, optimizer_class, initial_lr, model, **kwargs):
        """
        Wrap a native pytorch optimizer as a babilim optimizer.

        :param optimizer_class: The class which should be wrapped (not an instance).
         For example "optimizer_class=torch.optim.SGD".
        :param model: The model that is used (instance of type IModel).
        :param kwargs: The arguments for the optimizer on initialization.
        """
        super().__init__(initial_lr)
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self.model = model
        self.optim = None

    @RunOnlyOnce
    def build(self, lr):
        self.optim = self.optimizer_class(self.model.trainable_variables_native, lr=lr, **self.kwargs)

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        self.build(self.lr)
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr
        self.optim.step()
