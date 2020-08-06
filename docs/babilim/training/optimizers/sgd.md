[Back to Overview](../../../README.md)

# babilim.training.optimizers.sgd

> Common stochastic gradient descent optimizer.

---
---
## *class* **SGD**(Optimizer)

Common stochastic gradient descent optimizer.

* **initial_lr**: The initial learning rate for the optimizer. Learning rates are updated in the optimizer via callbacks.
* **momentum**: Value between 0 and 1 representing the momentum of the old grads to keep.
* **dampening**: Value between 0 and 1 representing by how much the accumulated gradients should be dampened.
* **weight_decay**: Value between 0 and 1 representing by how much the new gradients should be reduced.
* **nesterov**: If nesterov momentum should be used.


---
### *def* **apply_gradients**(*self*, gradients: Iterable[ITensor], variables: Iterable[ITensor]) -> None

This method applies the gradients to variables.

* **gradients**: An iterable of the gradients.
* **variables**: An iterable of the variables to which the gradients should be applied (in the same order as gradients).
* **learning_rate**: The learning rate which is currently used.


