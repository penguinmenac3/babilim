import sys
import numpy as np
import babilim
from babilim import PYTORCH_BACKEND, TF_BACKEND, Tensor, ITensor

babilim.set_backend(TF_BACKEND if "tf" in sys.argv else PYTORCH_BACKEND)

arr = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float32)
arr2 = arr + arr

a = Tensor(data=arr.copy(), trainable=True)
a.assign(arr)
assert a.trainable == True

b = Tensor(data=arr.copy(), trainable=False)
b.assign(arr)
assert b.trainable == False

res = Tensor(data=arr2, trainable=False)

# Check if assign and back to numpy works.
assert (a.numpy() == arr).all()

# Check if equal works.
assert a == b
assert (a.numpy() == b.numpy()).all()

# Check if operations work
assert a + b == res

# Check if assignments work
tmp = a.value
a.assign(a+b)
assert Tensor(data=None, trainable=False, native=a.value) == Tensor(data=None, trainable=False, native=tmp)
assert a == res
a -= b
assert a == b
a += b
assert a == res

# Check if a is 2 * b (Note how only b * 2 and not 2 * b is allowed)
assert a == b * 2

# Now that a is 2 b check comparisons
assert a > b
assert b < a
assert a >= b
assert b <= a
assert a >= a
assert a <= b

print("Test succeeded")
