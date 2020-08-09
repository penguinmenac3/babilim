[Back to Overview](../../../README.md)

# babilim.data.datasets.coco

> An implementation to load the coco dataset into babilim.

---
---
## *class* **CocoDataset**(Dataset)

*(no documentation found)*

---
### *def* **getitem**(*self*, idx: int)

Implements the getitem required by the babilim.data.Dataset.

This function gets called when you do `feat, label = dataset[idx]`.

* **idx**: The index in the dataset.
* **returns**: A tuple (feat, label). The type of feat is `self.InputType` and the type of label is `self.OutputType`.


---
### *def* **get_by_sample_token**(*self*, sample_token: str)

Get a datapoint 


Example:
```python
import matplotlib.pyplot as plt

class CocoConfig(Config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 1

config = CocoConfig()
dataset = CocoDataset(config, os.path.join(os.environ["DATA_PATH"], "COCO"), SPLIT_TRAIN)
dataset_input, dataset_output = dataset[0]

plt.figure(figsize=(12,12))
plt.imshow(dataset_input.image)
plt.show()
```
![data](../../../../docs/jlabdev_images/d693c11a2468ec5096e6020c6146ab92.png)

