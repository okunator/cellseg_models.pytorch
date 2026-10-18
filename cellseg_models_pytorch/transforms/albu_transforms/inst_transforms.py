import numpy as np

from cellseg_models_pytorch.utils import fix_duplicates

from ...utils import binarize
from ..functional import (
    gen_contour_maps,
    gen_dist_maps,
    gen_flow_maps,
    gen_hv_maps,
    gen_omni_flow_maps,
    gen_stardist_maps,
    gen_weight_maps,
    smooth_distance,
)
from ._composition import OnlyInstMapTransform

__all__ = [
    "CellposeTransform",
    "HoverNetTransform",
    "OmniposeTransform",
    "StardistTransform",
    "SmoothDistTransform",
    "DistTransform",
    "ContourTransform",
    "EdgeWeightTransform",
    "BinarizeTransform",
]


class CellposeTransform(OnlyInstMapTransform):
    """Generate two flow channels that guide pixels toward each object.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``cellpose``.
        out_channels: Number of target channels, 2. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        """Generate two flow channels that guide pixels toward each object.

        Args:
            deduplicate: Retained for compatibility; this class currently does not apply
                duplicate-label correction.
        """
        super().__init__()
        self.name = "cellpose"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating flow map (2, H, W), with vertical then horizontal channels. H is height and W is width.
        """

        return gen_flow_maps(inst)


class HoverNetTransform(OnlyInstMapTransform):
    """Generate horizontal and vertical position maps inside each object.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``hovernet``.
        out_channels: Number of target channels, 2. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        """Generate horizontal and vertical position maps inside each object.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "hovernet"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating position maps (2, H, W), vertical then horizontal. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_hv_maps(inst)


class OmniposeTransform(OnlyInstMapTransform):
    """Generate two flow channels from smooth distances inside each object.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``omnipose``.
        out_channels: Number of target channels, 2. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        """Generate two flow channels from smooth distances inside each object.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "omnipose"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating flow map (2, H, W), with vertical then horizontal channels. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_omni_flow_maps(inst)


class StardistTransform(OnlyInstMapTransform):
    """Generate radial distances from each object pixel to its boundary.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``stardist``.
        out_channels: Number of target channels, n_rays. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, n_rays: int = 32, deduplicate: bool = True, **kwargs):
        """Generate radial distances from each object pixel to its boundary.

        Args:
            n_rays: Number of ray directions; match the network output channels.
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
            **kwargs: Additional arguments are accepted but unused.
        """
        super().__init__()
        self.name = "stardist"
        self.n_rays = n_rays
        self.out_channels = n_rays
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating ray-distance map (n_rays, H, W), in pixels. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_stardist_maps(inst, self.n_rays)


class SmoothDistTransform(OnlyInstMapTransform):
    """Generate a smooth distance-to-boundary target inside each object.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``smoothdist``.
        out_channels: Number of target channels, 1. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True):
        """Generate a smooth distance-to-boundary target inside each object.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "smoothdist"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating smooth-distance map (H, W). H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return smooth_distance(inst)


class DistTransform(OnlyInstMapTransform):
    """Generate a distance-to-boundary target inside each object.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``dist``.
        out_channels: Number of target channels, 1. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        """Generate a distance-to-boundary target inside each object.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "dist"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating distance map (H, W), using the functional generator defaults. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_dist_maps(inst)


class ContourTransform(OnlyInstMapTransform):
    """Mark the boundaries of objects in an instance mask.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``contour``.
        out_channels: Number of target channels, 1. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True):
        """Mark the boundaries of objects in an instance mask.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "contour"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Floating contour map (H, W), with boundary pixels marked. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_contour_maps(inst)


class EdgeWeightTransform(OnlyInstMapTransform):
    """Assign larger loss weights to gaps between nearby objects.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``edgeweight``.
        out_channels: Number of target channels, 1. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self, deduplicate: bool = True) -> None:
        """Assign larger loss weights to gaps between nearby objects.

        Args:
            deduplicate: Give disconnected objects separate labels before target generation.
                Enable when the same ID may identify disconnected objects.
        """
        super().__init__()
        self.name = "edgeweight"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Float64 boundary-weight map (H, W), not a contour label mask. H is height and W is width.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_weight_maps(inst)


class BinarizeTransform(OnlyInstMapTransform):
    """Convert instance labels to foreground and background.

    Call on an integer instance mask: 0 is background and positive labels identify
    objects. Use with ``ApplyEach`` after joint image/mask augmentation.

    Attributes:
        name: Target-key suffix, ``binary``.
        out_channels: Number of target channels, 1. Scalar maps are HW;
            ``ApplyEach(as_list=True)`` adds their final singleton channel.
    """

    def __init__(self):
        """Configure binary target generation."""
        super().__init__()
        self.name = "binary"
        self.out_channels = 1

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate this target from one instance mask.

        Args:
            inst: Integer NumPy mask (H, W). Zero is background; positive IDs identify
                individual objects. The input mask is not modified.
            **kwargs: Extra call arguments are accepted for composition and ignored.

        Returns:
            Uint8 mask (H, W): 0 for background and 1 for foreground. H is height and W is width.
        """
        return binarize(inst)
