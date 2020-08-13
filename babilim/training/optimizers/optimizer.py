# AUTOGENERATED FROM: babilim/training/optimizers/optimizer.ipynb

# Cell: 0
from typing import Iterable
from babilim.core.itensor import ITensor
from babilim.core.module import Module
from babilim.core import RunOnlyOnce


# Cell: 1
class Optimizer(Module):
    def __init__(self, initial_lr: float):
        """
        An optimizer base class.
        
        :param initial_lr: The initial learning rate for the optimizer. Learning rates are updated in the optimizer via callbacks.
        """
        super().__init__()
        self.lr = initial_lr
        
    def call(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        """
        Maps to `apply_gradients`.
        """
        self.apply_gradients(gradients, variables)

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        """
        This method applies the gradients to variables.

        :param gradients: An iterable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        """
        raise NotImplementedError("Apply gradients must be implemented by every optimizer.")


# Cell: 2
class NativePytorchOptimizerWrapper(Optimizer):
    def __init__(self, optimizer_class, initial_lr, **kwargs):
        """
        Wrap a native pytorch optimizer as a babilim optimizer.

        :param optimizer_class: The class which should be wrapped (not an instance).
         For example "optimizer_class=torch.optim.SGD".
        :param kwargs: The arguments for the optimizer on initialization.
        """
        super().__init__(initial_lr)
        self.optimizer_class = optimizer_class
        self.kwargs = kwargs
        self.optim = None

    @RunOnlyOnce
    def build(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]):
        """
        Build the optimizer. Automatically is called when apply_gradients is called for the first time.
        
        :param gradients: An iterable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        """
        self.optim = self.optimizer_class([var.native for var in variables], lr=self.lr, **self.kwargs)

    def apply_gradients(self, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None:
        """
        This method applies the gradients to variables.

        :param gradients: An iterable of the gradients.
        :param variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
        """
        self.build(gradients, variables)
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr
        self.optim.step()
