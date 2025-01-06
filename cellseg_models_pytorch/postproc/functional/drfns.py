"""
Imported from DRFNS repo. Code style mods.

https://github.com/PeterJackNaylor/DRFNS/

MIT License

Copyright (c) 2018 PeterJackNaylor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.segmentation as segm
from skimage import img_as_ubyte

from cellseg_models_pytorch.transforms import percentile_normalize99
from cellseg_models_pytorch.utils import binarize

__all__ = ["post_proc_drfns"]


def h_minima_reconstruction(inv_dist_map: np.ndarray, lamb: int = 7) -> np.ndarray:
    """Perform a H minimma reconstruction via an erosion method.

    Parameters
    ----------
        inv_dist_map : np.ndarray
            Inverse distance map. Shape: (H, W).
        lamb : int, default=7
            Intensity shift value lambda.

    Returns
    -------
        np.ndarray:
            H minima reconstruction from the inverse distance transform.
            Shape: (H, W).
    """

    def making_top_mask(x: np.ndarray, lamb: int = lamb) -> int:
        return min(255, x + lamb)

    # vectorize for performance
    find_minima = np.vectorize(making_top_mask)
    shift_inv_dist_map = find_minima(inv_dist_map)

    # reconstruct
    seed = shift_inv_dist_map
    mask = inv_dist_map
    reconstructed = morph.reconstruction(seed, mask, method="erosion").astype("uint8")

    return reconstructed


def find_maxima(inv_dist_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Find all local maxima from 2D image.

    Parameters
    ----------
        inv_dist_map : np.ndarray
            Inverse distance map. Shape (H, W).
        mask : np.ndarray, default=None
            Binary mask to remove small debris. Shape (H, W).

    Returns
    -------
        np.ndarray:
            The found maxima. Shape (H, W).

    """
    reconstructed = h_minima_reconstruction(inv_dist_map, 40)

    res = reconstructed - inv_dist_map
    if mask is not None:
        res[mask == 0] = 0

    return res


def dynamic_ws_alias(
    dist_map: np.ndarray, binary_mask: np.ndarray, thresh: float = 0.5
) -> np.ndarray:
    """Run the dynamic watershed segmentation.

    Minor mods made. (Removed the suspicious stuff from the end.)

    Parameters
    ----------
        dist_map : np.ndarray
            A distance transform. Shape (H, W)
        binary_mask : np.ndarray
            A binary mask. Shape (H, W)
        thresh : float, default=0.5
            The threshold value to find markers from the `dist_map`.

    Returns
    -------
        np.ndarray:
            The labelled instance segmentation result. Shape (H, W).
    """
    # binarize probs and dist map
    binary_dist_map = dist_map > thresh

    # invert distmap
    inv_dist_map = 255 - img_as_ubyte(dist_map)

    # find markers from minima erosion reconstructed maxima of inv dist map
    reconstructed = h_minima_reconstruction(inv_dist_map)
    markers = find_maxima(reconstructed, mask=binary_dist_map)
    markers = ndi.label(markers)[0]

    # apply watershed
    ws = segm.watershed(reconstructed, markers, mask=binary_mask)

    return ws


def post_proc_drfns(
    inst_map: np.ndarray, dist_map: np.ndarray, thresh: float = 0.5, **kwargs
) -> np.ndarray:
    """DRFNS post processing pipeline.

    https://ieeexplore.ieee.org/document/8438559

    Slightly modified. Uses the thresholded prob_map as the mask param
    in watershed. Markers are computed from the regressed distance map
    (inverted).

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled or binary mask. Shape (H, W).
        dist_map : np.ndarray
            Distance transform. Shape (H, W)
        thresh : float, default=0.5
            threshold value for markers and binary mask

    Returns
    -------
        np.ndarray:
            The instance labelled segmentation mask. Shape (H, W)
    """
    dist_map = percentile_normalize99(dist_map, amin=0, amax=1)
    binary_mask = binarize(inst_map)
    result = dynamic_ws_alias(dist_map, binary_mask, thresh)

    return result
