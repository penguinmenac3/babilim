Babilim - Deep Learning across Pytorch and TF2
==============================================

TODO Teaser Image with Logo of babilim.

What is Babilim?
----------------

Babilim is a Deep Learning Framework designed for ease of use like Keras.
The API is designed to be intuitive and easy while allowing for complex production or research models.
On top of that babilim runs on top of Tensorflow 2 or Pytorch, whichever you prefer.

Babilim was designed for:

* Intuitive ease of use,
* a unified development experience across pytorch and tf2,
* flexibility for research and robustness for production


Selecting your backend
----------------------

Since babilim is designed for multiple backends (tensorflow 2 and pytorch), it gives you the choice which should be used.
When your company/university uses one of the two frameworks you are fine to use or you can follow your personal preference for private projects.

.. code-block:: python

    import babilim
    babilim.set_backend(babilim.PYTORCH_BACKEND)
    # or
    babilim.set_backend(babilim.TF_BACKEND)


Design Principles
-----------------

TODO

Tutorials & Examples
--------------------

TODO

Why called babilim?
-------------------

**TL;DR** Referencing to Tower of Babel.

The Tower of Babel narrative in Genesis 11:1-9 is a myth about mankind building a tower that could reach into the heavens.
However, the attempt gets set back and fails because they started speaking different languages and were no longer able to understand each other.

Luckily for AI development there is only two major frameworks which share nearly all market share.
Babilim is an attempt to unite the two to talk in a language compatible with both.

.. toctree::
   :caption: Getting Started
   :includehidden:

   tutorials.install

.. toctree::
   :caption: Packages
   :hidden:

   babilim
   babilim.callbacks
   babilim.core
   babilim.data
   babilim.experiment
   babilim.layers
   babilim.logger
   babilim.losses
   babilim.models
   babilim.optimizers
   babilim.utils

.. toctree::
   :caption: Examples
   :hidden:

   examples.mnist
   examples.mnist.pytorch

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

