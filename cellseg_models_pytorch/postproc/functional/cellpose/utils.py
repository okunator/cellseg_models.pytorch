"""
Imported from the cellpose repo. Minor code style mods for readability.

Copyright © 2020 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer. Redistributions in binary
form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials provided
with the distribution. Neither the name of HHMI nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, find_objects
from skimage.color import hsv2rgb

from cellseg_models_pytorch.transforms import (
    percentile_normalize,
    percentile_normalize99,
)

__all__ = ["gen_flows", "fill_holes_and_remove_small_masks", "normalize_field"]


def normalize_field(mu: np.ndarray) -> np.ndarray:
    """Normalize the flow field.

    Parameters
    ----------
        mu : np.ndarray
            The un-normalized y- and x- flows. Shape (2, H, W).

    Returns
    -------
        np.ndarray:
            The normalized y- and x- flows. Shape (2, H, W).
    """
    mag = np.sqrt(np.nansum(mu**2, axis=0))
    mask = np.logical_and(mag != 0, ~np.isnan(mag))
    out = np.zeros_like(mu)
    out = np.divide(mu, mag, out=out, where=mask)

    return out


def gen_flows(hover: np.ndarray) -> np.ndarray:
    """Convert Horizontal and Vertical gradients to cellpose flows.

    Parameters
    ----------
        flows : np.ndarray
            Horizontal and Vertical flows. Shape: (2, H, W).

    Returns
    -------
        np.ndarray:
            The optical flow representation. Shape (H, W, 3).
    """
    enhanced = percentile_normalize99(hover, amin=-1, amax=1)
    H = (np.arctan2(enhanced[0], enhanced[1]) + np.pi) / (2 * np.pi)
    S = percentile_normalize(enhanced[0] ** 2 + enhanced[1] ** 2)
    HSV = np.stack([H, S, S], axis=-1)
    HSV = np.clip(HSV, a_min=0.0, a_max=1.0)
    flow = (hsv2rgb(HSV) * 255).astype(np.uint8)

    return flow


def fill_holes_and_remove_small_masks(
    inst_map: np.ndarray, min_size: int = 15
) -> np.ndarray:
    """Fill holes in inst_map and discard objects smaller than `min_size`.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        min_size : int, default=15
            Minimum number of pixels per mask, can turn off with -1.

    Returns
    -------
        np.ndarray:
            Processed iinstance labelled mask. Shape (H, W).

    Raises
    ------
        ValueError: If input has wrong shape.
    """
    if inst_map.ndim != 2:
        raise ValueError(f"`inst_map` shape need to be 2D. Got {inst_map.shape}.")

    j = 0
    slices = find_objects(inst_map)
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = inst_map[slc] == (i + 1)
            npix = msk.sum()

            if min_size > 0 and npix < min_size:
                inst_map[slc][msk] = 0
            else:
                msk = binary_fill_holes(msk)
                inst_map[slc][msk] = j + 1
                j += 1

    return inst_map
