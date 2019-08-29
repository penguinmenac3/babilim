# Design Principles of Babilim

This document contains all core design principles for the babilim library.

* Ease of use (keras like)
* Not oppinionated (only weakly oppinionated)
* Framework agnostic
* Framework native as first class citizen

## Ease of use

Babilim should be as easy to use as keras. A network should be definable and trainable with just a couple of lines of code.
Training should be done by creating a model, adding a loss and optimizer and then fitting it to a dataset.

Somewhat like:
```python
model = MyClassifier()

# single output model
model.loss = "crossentropy"
# or multiple outputs as tuple/list
model.loss = ["crossentropy", "mse"]
# or multiple outputs as dict
model.loss = {"logits": "crossentropywithlogits", "position": "mse"}

model.optimizer = "adam"
model.fit(dataset)
```

You can either define a model by subclassing or using a `Sequential`.
```python
# As a sequential
model = Sequential()
model.add(Linear(...))
model.add(Relu(...))
model.add(Linear(...))
model.add(Relu(...))

# Or by subclassing a model (a layer should work the same way)
class MyModel(Model):
    def __init__(self, ...):
        # Initialize everything independant of input shape.
        self.linear1 = Linear(...)
        ...

    @RunOnlyOnce
    def build(self, feature_a: ITensor, feature_b: ITensor) -> None:
        # Initialize everything dependant on input shape.
        ...

    def call(self, feature_a: ITensor, feature_b: ITensor) -> Tuple[ITensor, ITensor]:
        self.build(feature_a, feature_b)  # Only called in case your model has not been build yet.
        ...
        return result_a, result_b
model = MyModel(...)
```

## Not oppinionated

Babilim provides some functionality without specifying how you must implement or use your layers and models. You can use the build in training functions, however, you can use your own as well.
You should be able to write your code how you want to.

An example would be allowing, multiple return styles for layers and models (single, tuple, list or dict) so you can find your style. Also we do not want to define how you input data in your layers, as a single value, tuple list or dict, your choice.

## Framework agnostic

The implementation should allow to write nearly all code without bothering about the underlying framework.
Once you write your model, it should be babilim.switch_backend(...) and you are using another framework.
No code changes required.
This however means, that for custom layers you will need to write them either using the ITensor interface or provide implementations for all frameworks.

## Framework native as first class citizen

The babilim api should encourage you to use it because you want to.
However, it should be your choice and not forced upon you.
Therefor, all layers and models should be callable as if they ware native in the respective framework.

For example a user should be able to call a layer on a `babilim.ITensor`, a `tensorflow.Tensor` or a `pytorch.Tensor` and the return type will be of the same type as they passed in.
Babilim automatically handles wrapping and unwrapping when calling a layer/model.

This gives you the oportunity to implement your code using babilim and later use it in a codebase which is tensorflow or pytorch only.

Converting a `babilim.ITensor` into a framework native tensor should be as easy as reading the `native` attribute of it.
