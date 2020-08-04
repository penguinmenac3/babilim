import torch
from torch.nn.functional import max_pool2d as _MaxPooling2D
from torch.nn.functional import max_pool1d as _MaxPooling1D
from torch.nn.functional import avg_pool2d as _AveragePooling2D
from torch.nn.functional import avg_pool1d as _AveragePooling1D
from torchvision.ops import roi_pool as _roi_pool
from torchvision.ops import roi_align as _roi_align


from babilim.core.module import Module
from babilim.core.tensor_pt import Tensor
from babilim.core.annotations import RunOnlyOnce
from babilim.model.modules.pt.flatten import Flatten


class MaxPooling2D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling2D(features.native, (2, 2)))


class MaxPooling1D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return Tensor(native=_MaxPooling1D(features.native, 2))

class GlobalMaxPooling2D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features: Tensor):
        shape = features.shape[1:2]
        print(shape)
        return Tensor(native=_MaxPooling2D(features.native, shape))

class GlobalMaxPooling1D(Module):
    def __init__(self):
        super().__init__()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features: Tensor):
        shape = features.shape[1]
        return Tensor(native=_MaxPooling1D(features.native, shape))

class GlobalAveragePooling2D(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return self.flatten(Tensor(native=_AveragePooling2D(features.native, features.native.size()[2:])))


class GlobalAveragePooling1D(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features):
        return self.flatten(Tensor(native=_AveragePooling1D(features.native, features.native.size()[2:])))


def _convert_boxes_to_roi_format(boxes):
    """
    Convert rois into the torchvision format.

    :param boxes: The roi boxes as a native tensor[B, K, 4].
    :return: The roi boxes in the format that roi pooling and roi align in torchvision require. Native tensor[B*K, 5].
    """
    concat_boxes = boxes.view((-1, 4))
    ids = torch.full_like(boxes[:, :, :1], 0)
    for i in range(boxes.shape[0]):
        ids[i, :, :] = i
    ids = ids.view((-1, 1))
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


class ROIPooling(Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features, rois):
        torchvision_rois = _convert_boxes_to_roi_format(rois.native)

        # :param aligned: (bool) If False, use the legacy implementation.
        #    If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
        #    This version in Detectron2
        result = _roi_pool(features.native, torchvision_rois, self.output_size, self.spatial_scale)

        # Fix output shape
        N, C, _, _ = features.native.shape
        result.view((N, -1, C, self.output_size[0], self.output_size[1]))
        return Tensor(native=result)


class ROIAlign(Module):
    def __init__(self, output_size, spatial_scale):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    @RunOnlyOnce
    def build(self, features):
        pass

    def call(self, features, rois):
        torchvision_rois = _convert_boxes_to_roi_format(rois.native)

        # :param aligned: (bool) If False, use the legacy implementation.
        #    If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
        #    This version in Detectron2
        result = _roi_align(features.native, torchvision_rois, self.output_size, self.spatial_scale, aligned=True)

        # Fix output shape
        N, C, _, _ = features.native.shape
        result.view((N, -1, C, self.output_size[0], self.output_size[1]))
        return Tensor(native=result)
