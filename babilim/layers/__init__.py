from babilim.layers.ilayer import ILayer
from babilim.layers.layers import Flatten, Linear, Sequential, Lambda, Conv2D, BatchNormalization, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Activation

MaxPool1D = MaxPooling1D
MaxPool2D = MaxPooling2D
Dense = Linear

__all__ = ["ILayer", "Flatten", "Linear", "Dense", "Sequential", "Lambda", "Conv2D", "BatchNormalization", "MaxPool1D", "MaxPool2D", "MaxPooling2D", "MaxPooling1D", "GlobalAveragePooling2D", "Activation"]
