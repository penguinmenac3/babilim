[Back to Overview](../../README.md)

# babilim.core.gradient_tape

> Keeps track of gradiens in a block.

This code is under the MIT License.

# Gradient Tape

The gradient tape helps with keeping track of the gradients with a unified API for pytorch and tensorflow.

### *def* **GradientTape**(variables: List) -> object

Collect the gradients for the block within a with statement.

* variables: The variables for which the gradients should be tracked.


A simple example illustrates the usage best.

```python
with GradientTape(model.trainable_variables) as tape:
    preds = model(**features._asdict())
    loss = loss_fn(preds, labels)
tape.gradient(loss)
```

