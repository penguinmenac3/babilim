[Back to Overview](../../README.md)

# babilim.training.losses

> A package containing all losses.

# *class* **Loss**(Module)

A loss is a statefull object which computes the difference between the prediction and the target.

* log_std: When true the loss will log its standard deviation. (default: False)
* log_min: When true the loss will log its minimum values. (default: False)
* log_max: When true the loss will log its maximal values. (default: False)


### *def* **call**(*self*, y_pred: Any, y_true: Any) -> ITensor

Implement a loss function between preds and true outputs.

DO NOT:
* Overwrite this function (overwrite `self.loss(...)` instead)
* Call this function (call the module instead `self(y_pred, y_true)`)

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


### *def* **loss**(*self*, y_pred: Any, y_true: Any) -> ITensor

Implement a loss function between preds and true outputs.

**`loss` must be overwritten by subclasses.**

DO NOT:
* Call this function (call the module instead `self(y_pred, y_true)`)

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


### *def* **log**(*self*, name: str, value: ITensor) -> None

Log a tensor under a name.

These logged values then can be used for example by tensorboard loggers.

* name: The name under which to log the tensor.
* value: The tensor that should be logged.


### *def* **reset_avg**(*self*) -> None

Reset the accumulation of tensors in the logging.

Should only be called by a tensorboard logger.


### *def* **summary**(*self*, samples_seen, **summary**_writer=None) -> None

Write a summary of the accumulated logs into tensorboard.

* samples_seen: The number of samples the training algorithm has seen so far (not iterations!).
This is used for the x axis in the plot. If you use the samples seen it is independant of the batch size.
If the network was trained for 4 batches with 32 batch size or for 32 batches with 4 batchsize does not matter.
* summary_writer: The summary writer to use for writing the summary. If none is provided it will use the tensorflow default.


### *def* **avg**(*self*)

Get the average of the loged values.

This is helpfull to print values that are more stable than values from a single iteration.


# *class* **NativeLossWrapper**(Loss)

Wrap a native loss as a babilim loss.

The wrapped object must have the following signature:

```python
Callable(y_pred, y_true, log_val) -> Tensor
```

where log_val will be a function which can be used for logging scalar tensors/values.

* loss: The loss that should be wrapped.
* log_std: When true the loss will log its standard deviation. (default: False)
* log_min: When true the loss will log its minimum values. (default: False)
* log_max: When true the loss will log its maximal values. (default: False)


### *def* **loss**(*self*, y_pred: Any, y_true: Any) -> ITensor

Compute the loss using the native loss function provided in the constructor.

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


# *class* **SparseCrossEntropyLossFromLogits**(Loss)

Compute a sparse cross entropy.

This means that the preds are logits and the targets are not one hot encoded.

* log_std: When true the loss will log its standard deviation. (default: False)
* log_min: When true the loss will log its minimum values. (default: False)
* log_max: When true the loss will log its maximal values. (default: False)


### *def* **loss**(*self*, y_pred: ITensor, y_true: ITensor) -> ITensor

Compute the sparse cross entropy assuming y_pred to be logits.

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.


# *class* **MeanSquaredError**(Loss)

Compute the mean squared error.

* log_std: When true the loss will log its standard deviation. (default: False)
* log_min: When true the loss will log its minimum values. (default: False)
* log_max: When true the loss will log its maximal values. (default: False)


### *def* **loss**(*self*, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor

Compute the mean squared error.

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* axis: (Optional) The axis along which to compute the mean squared error.


# *class* **SparseCategoricalAccuracy**(Loss)

Compute the sparse mean squared error.

Sparse means that the targets are not one hot encoded.

* log_std: When true the loss will log its standard deviation. (default: False)
* log_min: When true the loss will log its minimum values. (default: False)
* log_max: When true the loss will log its maximal values. (default: False)


### *def* **loss**(*self*, y_pred: ITensor, y_true: ITensor, axis: int=-1) -> ITensor

Compute the sparse categorical accuracy.

* y_pred: The predictions of the network. Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* y_true: The desired outputs of the network (labels). Either a NamedTuple pointing at ITensors or a Dict or Tuple of ITensors.
* axis: (Optional) The axis along which to compute the sparse categorical accuracy.


