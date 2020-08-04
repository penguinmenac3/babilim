import torch

from babilim.core import Module, RunOnlyOnce
from babilim.core.tensor_pt import Tensor


class Gather(Module):
    def __init__(self):
        super().__init__()

    def call(self, input_tensor, indices):
        # Unwrap variables and check if batch size matches.
        input_tensor = input_tensor.native
        indices = indices.native
        assert input_tensor.shape[0] == indices.shape[0]

        # Then gather the indices along the batches.
        result = torch.stack([torch.index_select(input_tensor[i], 0, indices[i]) for i in range(indices.shape[0])])
        return Tensor(native=result)


class TopKIndices(Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, input_tensor):
        result = torch.topk(input_tensor.native, self.k).indices
        return Tensor(native=result)
