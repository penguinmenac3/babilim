# babilim.model.modules.layers

> Here you can find all layers implemented in babilim.

# Various (Conv, Linear, BatchNorm)

### *def* **Flatten**() -> Module

Flatten a feature map into a linearized tensor.

This is usefull after the convolution layers before the dense layers. The (B, W, H, C) tensor gets converted ot a (B, N) tensor.

* returns: A module implementing the flatten layer.


### *def* **Linear**(out_features: int, activation=None) -> Module

A simple linear layer.

It computes Wx+b with no activation funciton.

* out_features: The number of output features.
* activation: The activation function that should be added after the linear layer.
* returns: A module implementing the linear layer.


### *def* **Dense**(out_features: int, activation=None) -> Module

A simple dense layer (alias for Linear Layer).

It computes Wx+b with no activation funciton.

* out_features: The number of output features.
* activation: The activation function that should be added after the dense layer.
* returns: A module implementing the dense layer.


### *def* **Conv1D**(filters: int, kernel_size: int, padding: Optional[str] = None, strides: int = 1, dilation_rate: int = 1, kernel_initializer: Optional[Any] = None, activation=None) -> Module

A 1d convolution layer.

* filters: The number of filters in the convolution. Defines the number of output channels.
* kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically 1, 3 or 5 are recommended.
* padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* stride: The offset between two convolutions that are applied. Typically 1. Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* dilation_rate: The dilation rate for a convolution.
* kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
* activation: The activation function that should be added after the dense layer.
* returns: A module implementing the convolution layer.


### *def* **Conv2D**(filters: int, kernel_size: Tuple[int, int], padding: Optional[str] = None, strides: Tuple[int, int] = (1, 1), dilation_rate: Tuple[int, int] = (1, 1), kernel_initializer: Optional[Any] = None, activation=None) -> Module

A 2d convolution layer.

* filters: The number of filters in the convolution. Defines the number of output channels.
* kernel_size: The kernel size of the convolution. Defines the area over which is convolved. Typically (1,1) (3,3) or (5,5) are recommended.
* padding: What type of padding should be applied. The string "none" means no padding is applied, None or "same" means the input is padded in a way that the output stays the same size if no stride is applied.
* stride: The offset between two convolutions that are applied. Typically (1, 1). Stride affects also the resolution of the output feature map. A stride 2 halves the resolution, since convolutions are only applied every odd pixel.
* dilation_rate: The dilation rate for a convolution.
* kernel_initializer: A kernel initializer function. By default orthonormal weight initialization is used.
* activation: The activation function that should be added after the dense layer.
* returns: A module implementing the convolution layer.


### *def* **BatchNormalization**() -> Module

A batch normalization layer.

* returns: A module implementing the batch normalization layer.


# Pooling

### *def* **GlobalMaxPooling2D**() -> Module

A global max pooling layer.

This computes the global max in W, H dimension, so that the result is of shape (B, C).

* returns: A module implementing the global max pooling 2d.


### *def* **GlobalMaxPooling1D**() -> Module

A global max pooling layer.

This computes the global max in N dimension (B, N, C), so that the result is of shape (B, C).

* returns: A module implementing the global max pooling 1d.


### *def* **MaxPooling2D**() -> Module

A 2x2 max pooling layer.

Computes the max of a 2x2 region with stride 2.
This halves the feature map size.

:return A module implementing the 2x2 max pooling.


### *def* **MaxPooling1D**() -> Module

A max pooling layer.

Computes the max of a 2 region with stride 2.
This halves the feature map size.

:return A module implementing the 2 max pooling.


### *def* **GlobalAveragePooling2D**() -> Module

A global average pooling layer.

This computes the global average in W, H dimension, so that the result is of shape (B, C).

* returns: A module implementing the global average pooling 2d.


### *def* **GlobalAveragePooling1D**() -> Module

A global average pooling layer.

This computes the global average in N dimension (B, N, C), so that the result is of shape (B, C).

* returns: A module implementing the global average pooling 1d.


# Activation Functions

### *def* **Activation**(activation: str) -> Module

Supports the activation functions.

* activation: A string specifying the activation function to use. (Only "relu" and None supported yet.)


# Composite Layers

### *def* **Lambda**(native_module, to_gpu=True) -> Module

Wrap a natively implemented layer into a babilim layer.

This can be used to implement layers that are missing in babilim in an easy way.

```
my_max = Lambda(tf.max)
```

* native_module: The native pytorch/tensorflow module that should be wrapped.
* to_gpu: (Optional) True if the module should be automatically be moved to the gpu. (default: True)


### *def* **Sequential**(*layers) -> Module

Create a module which is a sequential order of other layers.

Runs the layers in order.

```python
my_seq = Sequential(layer1, layer2, layer3)
```

* layers: All ordered parameters are used as layers.


