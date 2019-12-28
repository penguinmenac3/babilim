from babilim.model.modules.layers import Flatten, Linear, Sequential, Lambda, Conv2D, Conv1D, BatchNormalization, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Activation, GlobalMaxPooling2D, GlobalMaxPooling1D

MaxPool1D = MaxPooling1D
MaxPool2D = MaxPooling2D
Dense = Linear

__all__ = ["Flatten", "Linear", "Dense", "Sequential", "Lambda", "Conv2D", "Conv1D", "BatchNormalization", "MaxPool1D", "MaxPool2D", "MaxPooling2D", "MaxPooling1D", "GlobalAveragePooling2D", "Activation", "GlobalMaxPooling2D", "GlobalMaxPooling1D"]
