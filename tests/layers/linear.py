import sys
import numpy as np
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND, Tensor, ITensor
from babilim.layers import get_layer, Linear

babilim.set_backend(TF_BACKEND if "tf" in sys.argv else PYTORCH_BACKEND)
print(babilim.get_backend())

arr = np.array([[1,2,3, 4,5,6,7,8,9]], dtype=np.float32)

# Test a randomly initialized one.
a = Tensor(data=arr.copy(), trainable=True)
layer = Linear(1)
res = layer(a)
assert layer.trainable_variables[0] == layer.weight
assert layer.trainable_variables[1] == layer.bias

# Test initialized with numpy.
W = np.ones((1,9), dtype=np.float32)
b = np.zeros((1,), dtype=np.float32)
layer = Linear(1)
layer.build(a)
layer.weight.assign(W)
layer.bias.assign(b)
res = layer(a)
assert res.numpy()[0][0] == 45
assert (layer.trainable_variables[0].numpy() == W).all()
assert (layer.trainable_variables[1].numpy() == b).all()


# Test get_layer api
W = np.ones((1,9), dtype=np.float32)
b = np.zeros((1,), dtype=np.float32)
layer = get_layer("Linear")(1)
layer.build(a)
layer.weight.assign(W)
layer.bias.assign(b)
res = layer(a)
assert res.numpy()[0][0] == 45
assert (layer.trainable_variables[0].numpy() == W).all()
assert (layer.trainable_variables[1].numpy() == b).all()


print("Test succeeded")
