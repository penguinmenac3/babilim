# Babilim

> An attepmt to make a keras like framework for pytorch and tensorflow.


Read the [Documentation](https://penguinmenac3.github.io/babilim/index.html).

## What is Babilim?

Babilim is a Deep Learning Framework designed for ease of use like Keras. The API is designed to be intuitive and easy while allowing for complex production or research models. On top of that babilim runs on top of Tensorflow 2 or Pytorch, whichever you prefer. Seamless integration with TF2 and Pytorch was a top priority.

Babilim is designed for:
* Intuitive usage,
* a unified development experience across pytorch and tf2,
* flexibility for research and robustness for production.

## Install

Follow the official instructions [here](https://pytorch.org/get-started/locally/) to install pytorch or [here](https://www.tensorflow.org/install) to install tensorflow 2.

```
# Stable Version
pip install babilim

# Bleeding Edge
pip install git+https://github.com/penguinmenac3/babilim.git
```

## Selecting your Backend

Since babilim is designed for multiple backends (tensorflow 2 and pytorch), it gives you the choice which should be used. When your company/university uses one of the two frameworks you are fine to use or you can follow your personal preference for private projects.

```python
import babilim
babilim.set_backend(babilim.PYTORCH_BACKEND)
# or
babilim.set_backend(babilim.TF_BACKEND)
```

## Design Principles

Everything is attributed to one of three parts: Data, Model or Training. Some parts which are considered core functionality that is shared among them is in the core package.

* **Data** is concerned about loading and preprocessing the data for training, evaluation and deployment.
* **Model** is concerned with implementing the model. Everything required for the forward pass of the model is here.
* **Training** contains all required for training a model on data. This includes loss, metrics, optimizers and trainers.

## Tutorials & Examples

Starting with tutorials is usually easiest.
The tutorials do not focus on the shortest possible solution, but actually an overkill solution that shows as much as you would need to know for solving your own problem.

* [Fashion MNIST in Babilim](https://github.com/penguinmenac3/babilim/blob/master/examples/fashion_mnist.ipynb)
* Fashion MNIST with native pytorch [TODO]
* Fashion MNIST with native tensorflow 2 [TODO]

## Contributing

Currently there are no guidelines on how to contribute, so the best thing you can do is open up an issue and get in contact that way.
In the issue we can discuss how you can implement your new feature or how to fix that nasty bug.

## Why called babilim?

**TL;DR** Reference to Tower of Babel.

The Tower of Babel narrative in Genesis 11:1-9 is a myth about mankind building a tower that could reach into the heavens.
However, the attempt gets set back and fails because they started speaking different languages and were no longer able to understand each other.

Luckily for AI development there are only two major frameworks which share nearly all market share.
Babilim is an attempt to unite the two to talk in a language compatible with both.
