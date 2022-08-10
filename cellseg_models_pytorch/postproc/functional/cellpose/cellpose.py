"""
Imported from cellpose repo. Code style mods.

https://github.com/MouseLand/cellpose/blob/master/cellpose/dynamics.py

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
from scipy.ndimage.filters import maximum_filter1d
from skimage.filters import apply_hysteresis_threshold

from cellseg_models_pytorch.utils import binarize

from .integrator import follow_flows
from .utils import fill_holes_and_remove_small_masks, gen_flows

__all__ = ["post_proc_cellpose", "get_masks_cellpose"]


def get_masks_cellpose(
    p: np.ndarray,
    mask: np.ndarray,
    rpad: int = 20,
) -> np.ndarray:
    """Create masks using pixel convergence after running dynamics.

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------
        p : np.ndarray
            Final locations of each pixel after dynamics. Shape (2, H, W).
        mask : np.ndarray
            The binary mask of the cells. Shape (H, W).
        rpad : int, default=20
            Histogram edge padding.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W).
    """
    shape0 = p.shape[1:]
    dims = len(p)

    inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]), indexing="ij")

    for i in range(dims):
        p[i, ~mask] = inds[i][~mask]

    pflows = []
    edges = []
    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)
    shape = h.shape
    expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, None] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad

    # remove big masks
    M0 = M[tuple(pflows)]
    _, counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 1.0
    for i in np.nonzero(counts > big)[0]:
        M0[M0 == i] = 0

    _, M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    return M0


def post_proc_cellpose(
    inst_map: np.ndarray,
    flow_map: np.ndarray,
    dist_map: np.ndarray = None,
    return_flows: bool = False,
    min_size: int = 30,
    **kwargs
) -> np.ndarray:
    """Run the cellpose post-processing pipeline.

    https://www.nature.com/articles/s41592-020-01018-x

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled or binary mask. Shape (H, W).
        flow_map : np.ndarray
            Y- and x-flows. Shape: (2, H, W)
        dist_map : np.ndarray, default=None
            Regressed distance transform. Shape: (H, W).
        return_flows : bool, default=False
            If True, returns the HSV converted flows. They are just not
            needed for anything relevant.
        min_size : int
            The minimum size for the objects that will not be removed.

    Returns
    -------
        np.ndarray:
            The instance labelled segmentation mask. Shape (H, W)
    """
    #  convert channels to CHW
    if dist_map is not None:
        binary_mask = apply_hysteresis_threshold(dist_map, 0.5, 0.5)
    else:
        binary_mask = binarize(inst_map).astype(bool)

    dP = flow_map * binary_mask  # /5.
    # dP = normalize_field(dP)

    pixel_loc, _ = follow_flows(dP, niter=300, mask=binary_mask, suppress_euler=False)

    mask = get_masks_cellpose(p=pixel_loc, mask=binary_mask)
    inst_map = fill_holes_and_remove_small_masks(mask, min_size=min_size).astype("i4")

    if return_flows:
        hsv_flows = gen_flows(dP)
        return inst_map, hsv_flows

    return inst_map
