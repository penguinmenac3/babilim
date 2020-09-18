from babilim.core.tensor import TensorWrapper

_tensor_wrapper = TensorWrapper()


class GradientTapePT(object):
    def __init__(self, variables):
        self.variables = variables

    def __enter__(self):
        for var in self.variables:
            if var.native.grad is not None:
                var.native.grad.detach_()
                var.native.grad.zero_()
        return self

    def __exit__(self, type, value, traceback):
        pass

    def gradient(self, loss):
        loss.native.backward()
        gradients = [var.native.grad for var in self.variables]
        wrapped_grads = _tensor_wrapper.wrap(gradients)
        return wrapped_grads
