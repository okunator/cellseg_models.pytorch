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
    def __init__(self, deduplicate: bool = True) -> None:
        """Generate flows from a heat diffused label mask.

        https://www.nature.com/articles/s41592-020-01018-x

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "cellpose"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate flows.

        Parameters:
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Horizontal and vertical flows of objects.
                Shape: (2, H, W). Dtype: float64.
        """

        return gen_flow_maps(inst)


class HoverNetTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True) -> None:
        """Generate horizontal and vertical gradients from a label mask.

        https://www.sciencedirect.com/science/article/pii/S1361841519301045

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "hovernet"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate gradients.

        Parameters:
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Horizontal and vertical gradients of objects.
                Shape: (2, H, W). Dtype: float64.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_hv_maps(inst)


class OmniposeTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True) -> None:
        """Generate horizontal and vertical eikonal flows from a label mask.

        https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "omnipose"
        self.out_channels = 2
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate eikonal flows.

        Parameters:
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Horizontal and vertical gradients of objects.
                Shape: (2, H, W). Dtype: float64.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_omni_flow_maps(inst)


class StardistTransform(OnlyInstMapTransform):
    def __init__(self, n_rays: int = 32, deduplicate: bool = True, **kwargs):
        """Generate radial distance maps from a label mask.

        https://arxiv.org/abs/1806.03535

        Parameters:
            n_rays (int, default=32):
                Number of rays used for computing distance maps.
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "stardist"
        self.n_rays = n_rays
        self.out_channels = n_rays
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate radial distance maps.

        Parameters
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns
            np.ndarray:
                Pixelwise radial distance maps.
                Shape: (n_rays, H, W). Dtype: float64.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_stardist_maps(inst, self.n_rays)


class SmoothDistTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True):
        """Generate FIM distance transforms from a label mask.

        https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "smoothdist"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate smooth distance transforms.

        Parameters
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Smooth distance transforms of objects.
                Shape: (H, W). Dtype: float64.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return smooth_distance(inst)


class DistTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True) -> None:
        """Generate distance transforms from a label mask.

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "dist"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate distance transforms.

        Parameters
            inst (np.ndarray):
                Instance labelled mask. Shape: (H, W).

        Returns:
            np.ndarray:
                Distance transforms of objects.
                Shape: (H, W). Dtype: float64.
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_dist_maps(inst)


class ContourTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True):
        """Generate contour map from a label mask.

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "contour"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate contour transforms.

        Parameters
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Contour of objects. Shape: (H, W). Dtype: float64
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_contour_maps(inst)


class EdgeWeightTransform(OnlyInstMapTransform):
    def __init__(self, deduplicate: bool = True) -> None:
        """Generate weight maps for object boundaries.

        Parameters:
            deduplicate (bool, default=True):
                Whether to fix duplicate values in the mask before transforming.
                This adds overhead, so use only if you know that the mask has duplicates.
        """
        super().__init__()
        self.name = "edgeweight"
        self.out_channels = 1
        self.deduplicate = deduplicate

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate edge weight transforms.

        Parameters
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Contour of objects. Shape: (H, W). Dtype: float64
        """
        if self.deduplicate:
            inst = fix_duplicates(inst)
        return gen_weight_maps(inst)


class BinarizeTransform(OnlyInstMapTransform):
    def __init__(self):
        """Binarize instance labelled mask."""
        super().__init__()
        self.name = "binary"
        self.out_channels = 1

    def __call__(self, inst: np.ndarray, **kwargs) -> np.ndarray:
        """Generate a binary mask from instance labelled mask.

        Parameters:
            inst (np.ndarray):
                Instance labelled mask. Shape (H, W).

        Returns:
            np.ndarray:
                Binary mask. Shape: (H, W). Dtype: uint8
        """
        return binarize(inst)
