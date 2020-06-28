# babilim.training.optimizers.optimizer

> The optimizer interface.

# *class* **Optimizer**(StatefullObject)

An optimizer base class.

* initial_lr: The initial learning rate for the optimizer. Learning rates are updated in the optimizer via callbacks.


### *def* **apply_gradients**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

This method applies the gradients to variables.

* gradients: An iterable of the gradients.
* variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
* learning_rate: The learning rate which is currently used.


# *class* **NativePytorchOptimizerWrapper**(Optimizer)

Wrap a native pytorch optimizer as a babilim optimizer.

* optimizer_class: The class which should be wrapped (not an instance).
For example "optimizer_class=torch.optim.SGD".
* model: The model that is used (instance of type IModel).
* kwargs: The arguments for the optimizer on initialization.


### *def* **build**(*self*, lr)

Build the optimizer. Automatically is called when apply_gradients is called for the first time.

* lr: The learning rate used to initialize the optimizer.


### *def* **apply_gradients**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

This method applies the gradients to variables.

* gradients: An iterable of the gradients.
* variables: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
* learning_rate: The learning rate which is currently used.


