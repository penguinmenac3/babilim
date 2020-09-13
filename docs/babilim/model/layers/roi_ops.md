[Back to Overview](../../../README.md)

# babilim.model.layers.roi_ops

> Operations for region of interest extraction.



Convert rois into the torchvision format.

* **boxes**: The roi boxes as a native tensor[B, K, 4].
* **returns**: The roi boxes in the format that roi pooling and roi align in torchvision require. Native tensor[B*K, 5].


---
---
## *class* **RoiPool**(ModuleNative)

Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

Creates a callable object, when calling you can use these Arguments:
* **features**: (Tensor[N, C, H, W]) input tensor
* **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
* **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.

Parameters to RoiPool constructor:
* **output_size**: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
* **spatial_scale**: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

roi = RoiPool(output_size=(7, 4))
tensor = Tensor(data=np.zeros((2,3,24,24), dtype=np.float32), trainable=False)
rois = Tensor(data=np.array([[[0,0,12,12],[4,7,6,23]], [[0,0,12,12], [4,7,6,23]]], dtype=np.float32), trainable=False)

print(rois.shape)
print(tensor.shape)
result = roi(tensor, rois)
print(result.shape)
```
Output:
```
(2, 2, 4)
(2, 3, 24, 24)
(2, 2, 3, 7, 4)

```

---
---
## *class* **RoiAlign**(ModuleNative)

Performs Region of Interest (RoI) Align operator described in Mask R-CNN.

Creates a callable object, when calling you can use these Arguments:
* **features**: (Tensor[N, C, H, W]) input tensor
* **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
* **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.

Parameters to RoiAlign constructor:
* **output_size**: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
* **spatial_scale**: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0


Example:
```python
from babilim.core.tensor import Tensor
import numpy as np

roi = RoiAlign(output_size=(7, 4))
tensor = Tensor(data=np.zeros((2,3,24,24), dtype=np.float32), trainable=False)
rois = Tensor(data=np.array([[[0,0,12,12],[4,7,6,23]], [[0,0,12,12], [4,7,6,23]]], dtype=np.float32), trainable=False)

print(rois.shape)
print(tensor.shape)
result = roi(tensor, rois)
print(result.shape)
```
Output:
```
(2, 2, 4)
(2, 3, 24, 24)
(2, 2, 3, 7, 4)

```

