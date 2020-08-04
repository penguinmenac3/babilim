[Back to Overview](../../../README.md)

# babilim.training.optimizers.learning_rates

> A package with all typical learning rate schedules.

# *class* **LearningRateSchedule**(Module)

An interface to a learning rate schedule.

It must implement a `__call__(self, global_step: int) -> float` method which converts a global_step into the current lr.


# *class* **Const**(LearningRateSchedule)

A constant learning rate.

* lr: The learning rate that should be set.


# *class* **Exponential**(LearningRateSchedule)

Exponential learning rate decay.

lr = initial_lr * e^(-k * step)

* initial_lr: The learning rate from which is started.
* k: The decay rate.


# *class* **StepDecay**(LearningRateSchedule)

A steped decay.
Multiply the learning rate by `drop` every `steps_per_drop`.

* initial_lr: The learning rate with which should be started.
* drop: By what the learning rate is multiplied every steps_per_drop steps.
* steps_per_drop: How many steps should be done between drops.


