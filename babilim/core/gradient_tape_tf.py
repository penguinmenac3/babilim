from babilim.core.tensor import TensorWrapper
import tensorflow as tf

_tensor_wrapper = TensorWrapper()

class GradientTapeTF(object):
    def __init__(self, variables):
        self.variables = variables
        self.tape = tf.GradientTape()
        self.tape_context = None

    def __enter__(self):
        self.tape_context = self.tape.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.tape_context.__exit__(type, value, traceback)

    def gradient(self, loss):
        raw_vars = _tensor_wrapper.unwrap(self.variables)
        gradients = self.tape_context.gradient(loss.native, raw_vars)
        wrapped_grads = _tensor_wrapper.wrap(gradients)
        return wrapped_grads
