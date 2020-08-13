[Back to Overview](../../../README.md)

# babilim.training.optimizers.optimizer

> The optimizer interface.

---
---
## *class* **Optimizer**(Module)

An optimizer base class.

* **initial_lr**: The initial learning rate for the optimizer. Learning rates are updated in the optimizer via callbacks.


---
### *def* **call**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

Maps to `apply_gradients`.


---
### *def* **apply_gradients**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

This method applies the gradients to variables.

* **gradients**: An iterable of the gradients.
* **variables**: An iterable of the variables to which the gradients should be applied (in the same order as gradients).


---
---
## *class* **NativePytorchOptimizerWrapper**(Optimizer)

Wrap a native pytorch optimizer as a babilim optimizer.

* **optimizer_class**: The class which should be wrapped (not an instance).
For example "optimizer_class=torch.optim.SGD".
* **kwargs**: The arguments for the optimizer on initialization.


---
### *def* **build**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor])

Build the optimizer. Automatically is called when apply_gradients is called for the first time.

* **gradients**: An iterable of the gradients.
* **variables**: An iterable of the variables to which the gradients should be applied (in the same order as gradients).


---
### *def* **apply_gradients**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

This method applies the gradients to variables.

* **gradients**: An iterable of the gradients.
* **variables**: An iterable of the variables to which the gradients should be applied (in the same order as gradients).


