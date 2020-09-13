# AUTOGENERATED FROM: babilim/model/layers/roi_ops.ipynb

# Cell: 0
"""
# babilim.model.layers.roi_ops

> Operations for region of interest extraction.
"""

# Cell: 1
from babilim.core.annotations import RunOnlyOnce
from babilim.core.module_native import ModuleNative


# Cell: 2
def _convert_boxes_to_roi_format(boxes):
    """
    Convert rois into the torchvision format.

    :param boxes: The roi boxes as a native tensor[B, K, 4].
    :return: The roi boxes in the format that roi pooling and roi align in torchvision require. Native tensor[B*K, 5].
    """
    import torch
    concat_boxes = boxes.view((-1, 4))
    ids = torch.full_like(boxes[:, :, :1], 0)
    for i in range(boxes.shape[0]):
        ids[i, :, :] = i
    ids = ids.view((-1, 1))
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


# Cell: 3
class RoiPool(ModuleNative):
    def __init__(self, output_size, spatial_scale=1.0):
        """
        Performs Region of Interest (RoI) Pool operator described in Fast R-CNN.

        Creates a callable object, when calling you can use these Arguments:
        * **features**: (Tensor[N, C, H, W]) input tensor
        * **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
        * **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.
        
        Parameters to RoiPool constructor:
        :param output_size: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
        :param spatial_scale: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    @RunOnlyOnce
    def _build_pytorch(self, features, rois):
        pass
        
    def _call_pytorch(self, features, rois):
        from torchvision.ops import roi_pool as _roi_pool
        torchvision_rois = _convert_boxes_to_roi_format(rois)

        result = _roi_pool(features, torchvision_rois, self.output_size, self.spatial_scale)

        # Fix output shape
        N, C, _, _ = features.shape
        result = result.view((N, -1, C, self.output_size[0], self.output_size[1]))
        return result

    @RunOnlyOnce
    def _build_tf(self, features, rois):
        # TODO implement
        raise NotImplementedError()
        
    def _call_tf(self, features, rois):
        # TODO implement
        raise NotImplementedError()


# Cell: 4
class RoiAlign(ModuleNative):
    def __init__(self, output_size, spatial_scale=1.0):
        """
        Performs Region of Interest (RoI) Align operator described in Mask R-CNN.

        Creates a callable object, when calling you can use these Arguments:
        * **features**: (Tensor[N, C, H, W]) input tensor
        * **rois**: (Tensor[N, K, 4]) the box coordinates in (x1, y1, x2, y2) format where the regions will be taken from.
        * **return**: (Tensor[N, K, C, output_size[0], output_size[1]]) The feature maps crops corresponding to the input rois.
        
        Parameters to RoiAlign constructor:
        :param output_size: (Tuple[int, int]) the size of the output after the cropping is performed, as (height, width)
        :param spatial_scale: (float) a scaling factor that maps the input coordinates to the box coordinates. Default: 1.0
        """
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        
    @RunOnlyOnce
    def _build_pytorch(self, features, rois):
        pass
        
    def _call_pytorch(self, features, rois):
        from torchvision.ops import roi_align as _roi_align
        torchvision_rois = _convert_boxes_to_roi_format(rois)

        # :param aligned: (bool) If False, use the legacy implementation.
        #    If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
        #    This version in Detectron2
        result = _roi_align(features, torchvision_rois, self.output_size, self.spatial_scale, aligned=True)

        # Fix output shape
        N, C, _, _ = features.shape
        result = result.view((N, -1, C, self.output_size[0], self.output_size[1]))
        return result

    @RunOnlyOnce
    def _build_tf(self, features, rois):
        # TODO implement
        raise NotImplementedError()
        
    def _call_tf(self, features, rois):
        # TODO implement
        raise NotImplementedError()