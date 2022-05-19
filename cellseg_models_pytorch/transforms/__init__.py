from ._composition import apply_each, compose, to_tensor, to_tensorv3
from .img_transforms import (
    blur_transforms,
    center_crop,
    hue_saturation_transforms,
    non_rigid_transforms,
    non_spatial_transforms,
    random_crop,
    resize,
    rigid_transforms,
)
from .inst_transforms import (
    binarize_transform,
    cellpose_transform,
    contour_transform,
    dist_transform,
    edgeweight_transform,
    hovernet_transform,
    omnipose_transform,
    smooth_dist_transform,
)
from .norm_transforms import (
    imgnorm_transform,
    minmaxnorm_transform,
    percentilenorm_transform,
)

IMG_TRANSFORMS = {
    "blur": blur_transforms,
    "center_crop": center_crop,
    "hue_sat": hue_saturation_transforms,
    "non_rigid": non_rigid_transforms,
    "rigid": rigid_transforms,
    "non_spatial": non_spatial_transforms,
    "resize": resize,
    "random_crop": random_crop,
}

INST_TRANSFORMS = {
    "cellpose": cellpose_transform,
    "contour": contour_transform,
    "dist": dist_transform,
    "edgeweight": edgeweight_transform,
    "hovernet": hovernet_transform,
    "omnipose": omnipose_transform,
    "smooth_dist": smooth_dist_transform,
    "binarize": binarize_transform,
}

NORM_TRANSFORMS = {
    "norm": imgnorm_transform,
    "percentile": percentilenorm_transform,
    "minmax": minmaxnorm_transform,
}

__all__ = [
    "IMG_TRANSFORMS",
    "NORM_TRANSFORMS",
    "INST_TRANSFORMS",
    "rigid_transforms",
    "non_rigid_transforms",
    "non_spatial_transforms",
    "hue_saturation_transforms",
    "blur_transforms",
    "random_crop",
    "center_crop",
    "resize",
    "compose",
    "apply_each",
    "to_tensor",
    "to_tensorv3",
    "cellpose_transform",
    "hovernet_transform",
    "omnipose_transform",
    "dist_transform",
    "smooth_dist_transform",
    "contour_transform",
    "edgeweight_transform",
    "binarize_transform",
    "imgnorm_transform",
    "percentilenorm_transform",
    "minmaxnorm_transform",
]
