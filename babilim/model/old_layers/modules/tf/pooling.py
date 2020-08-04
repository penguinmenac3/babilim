from tensorflow.keras.layers import MaxPooling2D as _MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D as _MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D as _GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling1D as _GlobalAveragePooling1D

from babilim.core.module import Module
from babilim.core.tensor_tf import Tensor
from babilim.core.annotations import RunOnlyOnce


class MaxPooling2D(Module):
    def __init__(self):
        super().__init__()
        self.pool = _MaxPooling2D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))


class MaxPooling1D(Module):
    def __init__(self):
        super().__init__()
        self.pool = _MaxPooling1D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))


class GlobalAveragePooling2D(Module):
    def __init__(self):
        super().__init__()
        self.pool = _GlobalAveragePooling2D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))


class GlobalAveragePooling1D(Module):
    def __init__(self):
        super().__init__()
        self.pool = _GlobalAveragePooling1D()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=self.pool(features.native))


class ROIPooling(Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        raise NotImplementedError()


class ROIAlign(Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        raise NotImplementedError()
