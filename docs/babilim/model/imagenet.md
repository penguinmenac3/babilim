[Back to Overview](../../README.md)

# babilim.model.imagenet

> An implemntation of various imagenet models.

---
---
## *class* **ImagenetModel**(ModuleNative)

Create one of the iconic image net models in one line.
Allows for only using the encoder part.

This model assumes the input image to be 0-255 (8 bit integer) with 3 channels.

* **encoder_type**: The encoder type that should be used. Must be in ("vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "resnet50", "resnet101", "resnet152", "densenet121", "densenet169", "densenet201", "inception_v3", "mobilenet_v2")
* **only_encoder**: Leaves out the classification head for VGG16 leaving you with a feature encoder.
* **pretrained**: If you want imagenet weights for this network.


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

encoder = ImagenetModel("vgg16_bn", only_encoder=True, pretrained="imagenet")
fake_image_batch_pytorch = Tensor(data=np.zeros((1, 3, 256, 256), dtype=np.float32), trainable=False)
print(fake_image_batch_pytorch.shape)
result = encoder(fake_image_batch_pytorch)
print(result.shape)
```
Output:
```
(1, 3, 256, 256)
(1, 512, 8, 8)

```

Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

model = ImagenetModel("resnet50", only_encoder=False, pretrained="imagenet")
fake_image_batch_pytorch = Tensor(data=np.zeros((1, 3, 256, 256), dtype=np.float32), trainable=False)
print(fake_image_batch_pytorch.shape)
result = model(fake_image_batch_pytorch)
print(result.shape)
```
Output:
```
(1, 3, 256, 256)
Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to C:\Users\fuerst/.cache\torch\hub\checkpoints\resnet50-19c8e357.pth
100%|██████████| 97.8M/97.8M [00:23<00:00, 4.41MB/s]
(1, 1000)

```

