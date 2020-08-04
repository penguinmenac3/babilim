from babilim.core import Module, RunOnlyOnce
from babilim.core.tensor_tf import Tensor


class Gather(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        raise NotImplementedError()


class TopKIndices(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        raise NotImplementedError()
