import math


class LearningRateSchedule(object):
    """
    An interface to a learning rate schedule.
    It should implement a call method which converts a global_step into the current lr.
    """
    def __call__(self, global_step: int) -> float:
        raise NotImplementedError("Must be implemented by subclass.")


class Const(LearningRateSchedule):
    def __init__(self, lr: float):
        """
        A constant learning rate.
        
        :param lr: The learning rate that should be set.
        """
        self.lr = lr

    def __call__(self, global_step: int) -> float:
        return self.lr


class Exponential(LearningRateSchedule):
    def __init__(self, initial_lr: float, k: float):
        """
        Exponential learning rate decay.

        lr = initial_lr * e^(-k * step)
        
        :param initial_lr: The learning rate from which is started.
        :param k: The decay rate.
        """
        self.initial_lr = initial_lr
        self.k = k

    def __call__(self, global_step: int) -> float:
        return self.initial_lr * math.exp(-self.k * global_step)


class StepDecay(LearningRateSchedule):
    def __init__(self, initial_lr: float, drop: float, steps_per_drop: int):
        """
        A steped decay.
        Multiply the learning rate by `drop` every `steps_per_drop`.

        :param initial_lr: The learning rate with which should be started.
        :param drop: By what the learning rate is multiplied every steps_per_drop steps.
        :param steps_per_drop: How many steps should be done between drops.
        """
        self.initial_lr = initial_lr
        self.drop = drop
        self.steps_per_drop = steps_per_drop

    def __call__(self, global_step: int) -> float:
        return self.initial_lr * math.pow(self.drop, math.floor((1+global_step)/self.steps_per_drop))

