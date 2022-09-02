from typing import List

import numpy as np

from cellseg_models_pytorch.utils import fix_duplicates

from ...utils import binarize
from ..functional import (
    gen_contour_maps,
    gen_dist_maps,
    gen_flow_maps,
    gen_hv_maps,
    gen_omni_flow_maps,
    gen_radial_distmaps,
    gen_stardist_maps,
    gen_weight_maps,
    smooth_distance,
)
from ._composition import OnlyInstMapTransform

__all__ = [
    "cellpose_transform",
    "hovernet_transform",
    "omnipose_transform",
    "stardist_transform",
    "stardist_opt_transform",
    "dist_transform",
    "smooth_dist_transform",
    "edgeweight_transform",
    "contour_transform",
    "binarize_transform",
]


class CellposeTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate flows from a heat diffused label mask.

        https://www.nature.com/articles/s41592-020-01018-x
        """
        super().__init__()
        self.name = "cellpose"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate flows.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Horizontal and vertical flows of objects.
                Shape: (2, H, W). Dtype: float64.
        """
        return gen_flow_maps(fix_duplicates(inst_map))


class HoVerNetTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate horizontal and vertical gradients from a label mask.

        https://www.sciencedirect.com/science/article/pii/S1361841519301045
        """
        super().__init__()
        self.name = "hovernet"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate gradients.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Horizontal and vertical gradients of objects.
                Shape: (2, H, W). Dtype: float64.
        """
        return gen_hv_maps(fix_duplicates(inst_map))


class OmniposeTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate horizontal and vertical eikonal flows from a label mask.

        https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2
        """
        super().__init__()
        self.name = "omnipose"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate eikonal flows.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Horizontal and vertical gradients of objects.
                Shape: (2, H, W). Dtype: float64.
        """
        return gen_omni_flow_maps(fix_duplicates(inst_map))


# Not useful for now.
class StardistSelfTrans(OnlyInstMapTransform):
    def __init__(self, n_rays: int = 32, n_segments: int = 20):
        """Generate radial distance maps from a label mask.

        https://arxiv.org/abs/1806.03535

        Parameters
        ----------
            n_rays : int, default=32
                Number of rays used for computing distance maps.
            n_segments : int, default=20
                Number of line segments the contour is divided into.
                The more segments used, the more detail is preserved with
                performance tradeoff.
        """
        super().__init__()
        self.name = "stardist"
        self.n_rays = n_rays
        self.n_seg = n_segments

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate radial distance maps.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Pixelwise radial distance maps.
                Shape: (n_rays, H, W). Dtype: float64.
        """
        return gen_radial_distmaps(fix_duplicates(inst_map), self.n_rays, self.n_seg)


class StardistTrans(OnlyInstMapTransform):
    def __init__(self, n_rays: int = 32, **kwargs):
        """Generate radial distance maps from a label mask.

        https://arxiv.org/abs/1806.03535

        Parameters
        ----------
            n_rays : int, default=32
                Number of rays used for computing distance maps.
            n_segments : int, default=20
                Number of line segments the contour is divided into.
                The more segments used, the more detail is preserved with
                performance tradeoff.
        """
        super().__init__()
        self.name = "stardist"
        self.n_rays = n_rays

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Fix duplicate values and generate radial distance maps.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Pixelwise radial distance maps.
                Shape: (n_rays, H, W). Dtype: float64.
        """
        return gen_stardist_maps(fix_duplicates(inst_map), self.n_rays)


class SmoothDistTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate FIM distance transforms from a label mask.

        https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2
        """
        super().__init__()
        self.name = "smoothdist"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Generate smooth distance transforms.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Smooth distance transforms of objects.
                Shape: (H, W). Dtype: float64.
        """
        return smooth_distance(inst_map)


class DistTrans(OnlyInstMapTransform):
    def __init__(self) -> None:
        """Generate distance transforms from a label mask."""
        super().__init__()
        self.name = "dist"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Generate distance transforms.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape: (H, W).

        Returns
        -------
            np.ndarray:
                Distance transforms of objects.
                Shape: (H, W). Dtype: float64.
        """
        return gen_dist_maps(fix_duplicates(inst_map))


class ContourTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate contour map from a label mask."""
        super().__init__()
        self.name = "contour"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Generate contour transforms.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Contour of objects. Shape: (H, W). Dtype: float64
        """
        return gen_contour_maps(fix_duplicates(inst_map))


class EdgeWeightTrans(OnlyInstMapTransform):
    def __init__(self):
        """Generate weight maps for object boundaries."""
        super().__init__()
        self.name = "edgeweight"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Generate edge weight transforms.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Contour of objects. Shape: (H, W). Dtype: float64
        """
        return gen_weight_maps(inst_map)


class BinarizeTrans(OnlyInstMapTransform):
    def __init__(self):
        """Binarize instance labelled mask."""
        super().__init__()
        self.name = "binary"

    def apply_to_instmap(self, inst_map: np.ndarray, **kwargs) -> np.ndarray:
        """Generate a binary mask from instance labelled mask.

        Parameters
        ----------
            inst_map : np.ndarray
                Instance labelled mask. Shape (H, W).

        Returns
        -------
            np.ndarray:
                Binary mask. Shape: (H, W). Dtype: uint8
        """
        return binarize(inst_map)


def cellpose_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the Cellpose tranformation for label mask."""
    return [CellposeTrans()]


def hovernet_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the HoVerNet tranformation for label mask."""
    return [HoVerNetTrans()]


def omnipose_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the omnipose tranformation for label mask."""
    return [OmniposeTrans()]


def stardist_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the stardist tranformation for label mask."""
    return [StardistTrans(**kwargs)]


def stardist_opt_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the stardist tranformation for label mask."""
    return [StardistSelfTrans(**kwargs)]


def dist_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the distance tranformation for label mask."""
    return [DistTrans()]


def smooth_dist_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the smooth distance tranformation for label mask."""
    return [SmoothDistTrans()]


def contour_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the contour tranformation for label mask."""
    return [ContourTrans()]


def edgeweight_transform(**kwargs) -> List[OnlyInstMapTransform]:
    """Return the edge weight tranformation for label mask."""
    return [EdgeWeightTrans()]


def binarize_transform(**kwargs):
    """Return the binarization tranformation for label mask."""
    return [BinarizeTrans()]
