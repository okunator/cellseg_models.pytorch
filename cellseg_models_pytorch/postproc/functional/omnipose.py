"""
Imported from the cellpose repo. Code style mods.

https://github.com/MouseLand/cellpose/blob/master/cellpose/omnipose/utils.py
https://github.com/MouseLand/cellpose/blob/master/cellpose/omnipose/core.py

Copyright © 2020 Howard Hughes Medical Institute

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer. Redistributions in
binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution. Neither the name of HHMI
nor the names of its contributors may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
“AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

from cellseg_models_pytorch.transforms import percentile_normalize
from cellseg_models_pytorch.utils import binarize

from .cellpose.integrator import follow_flows
from .cellpose.utils import (
    fill_holes_and_remove_small_masks,
    gen_flows,
    normalize_field,
)

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    raise ImportError(
        """scikit-learn is required for omnipose. `pip install -U scikit-learn`"""
    )


__all__ = ["post_proc_omnipose", "div_rescale", "get_masks_omnipose"]


def div_rescale(dP: np.ndarray, mask: np.ndarray, pad: int = 1) -> np.ndarray:
    """Rescale the divergence of the regressed flows.

    Parameters
    ----------
        dP : np.ndarray
            The regressed eikonal flows. Shape (2, H, W).
        mask : np.ndarray
            Binary mask of the predicted cells. Shape (H, W).
        pad : int, default=1
            Number of padded pixels around the input.

    Returns
    -------
        np.ndarray:
            The rescaled eikonal flow-maps. Shape: (2, H, W).
    """
    dP = dP.copy()
    dP *= mask
    dP = normalize_field(dP)

    # compute the divergence
    y, x = np.nonzero(mask)
    H, W = mask.shape

    Tx = np.zeros((H + 2 * pad) * (W + 2 * pad), np.float64)
    Tx[y * W + x] = np.reshape(dP[1].copy(), H * W)[y * W + x]

    Ty = np.zeros((H + 2 * pad) * (W + 2 * pad), np.float64)
    Ty[y * W + x] = np.reshape(dP[0].copy(), H * W)[y * W + x]

    # Rescaling by the divergence
    div = np.zeros(H * W, np.float64)
    div[y * W + x] = (
        Ty[(y + 2) * W + x]
        + 8 * Ty[(y + 1) * W + x]
        - 8 * Ty[(y - 1) * W + x]
        - Ty[(y - 2) * W + x]
        + Tx[y * W + x + 2]
        + 8 * Tx[y * W + x + 1]
        - 8 * Tx[y * W + x - 1]
        - Tx[y * W + x - 2]
    )

    div.shape = (H, W)
    div = percentile_normalize(div)

    dP *= div

    return dP


def get_masks_omnipose(
    p: np.ndarray,
    mask: np.ndarray,
    inds: np.ndarray = None,
) -> np.ndarray:
    """Omnipose mask recontruction algorithm.

    Parameters
    ----------
        p : np.ndarray
            All the pixel locations for the pixels after running the
            euler integrator. Shape: (2, H, W). Dtype: float32.
        mask : np.ndarray
            The binary mask of the cells. Shape (H, W).
        inds : np.ndarray
            Indices of the non-zero pixels. Shape: (n non zero pxls, 2).

    Returns
    -------
        np.ndarray:
            The instance labelled mask. Shape (H, W).
    """
    eps = 1 + (1 / 3)

    newinds = p[:, inds[:, 0], inds[:, 1]].swapaxes(0, 1)
    mask = np.zeros((p.shape[1], p.shape[2]))

    try:
        db = DBSCAN(eps=eps, min_samples=3, n_jobs=1).fit(newinds)
        labels = db.labels_
        mask[inds[:, 0], inds[:, 1]] = labels + 1
    except Exception:
        pass

    return mask.astype("i4")


def post_proc_omnipose(
    inst_map: np.ndarray,
    flow_map: np.ndarray,
    return_flows: bool = False,
    min_size: int = 30,
    **kwargs,
) -> np.ndarray:
    """Run the omnipose post-processing pipeline.

    More info in the omnipose paper:
    https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled or binary mask. Shape (H, W).
        flow_map : np.ndarray
            Y- and x-flows. Shape: (2, H, W)
        return_flows : bool, default=False
            If True, returns the HSV converted flows. They are just not
            needed for anything relevant.
        min_size : int
            The minimum size for the objects that will not be removed.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W). Dtype: int32.
    """
    #  convert channels to CHW
    binary_mask = binarize(inst_map).astype(bool)

    dP = div_rescale(flow_map, binary_mask)
    pixel_loc, inds = follow_flows(dP, niter=200, mask=binary_mask, suppress_euler=True)

    mask = get_masks_omnipose(p=pixel_loc, mask=binary_mask, inds=inds)

    inst_map = fill_holes_and_remove_small_masks(mask, min_size=min_size).astype("i4")

    if return_flows:
        hsv_flows = gen_flows(dP)
        return inst_map, hsv_flows

    return inst_map
