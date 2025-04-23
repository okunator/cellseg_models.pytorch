"""
Imported from cellpose repo. Refacotred for readability.

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

from typing import List, Tuple

import numpy as np
from scipy.ndimage import maximum_filter1d

from cellseg_models_pytorch.utils import binarize

from .integrator import follow_flows
from .utils import fill_holes_and_remove_small_masks, gen_flows

__all__ = ["post_proc_cellpose", "get_masks_cellpose"]


def get_seeds(
    p: np.ndarray, rpad: int = 20, dims: int = 2
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, List[np.ndarray]]:
    """Get seed points to cells from the 2D-histogram of the flows.

    Return also the histogram and flows.

    Parameters
    ----------
        p : np.ndarray
            The computed flows. Shape (2, H, W). Dtype: float32.
        rpad : int, default=20
            The amount of padding when computing seeds.
        dims : int, default=2
            The number of dimensions in the input data.

    Returns
    -------
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, List[np.ndarray]]:
            seeds : Tuple[np.ndarray, np.ndarray]
                The y- and x- seeds sorted by descending bincount.
                np.ndarray.shape: (n_cells, ). Dtype: int32.
            h : np.ndarray
                The 2D histogram of the flows p. Shape (H+pad, W+pad).
            pflows : List[np.ndarray]
                The flattend y and x flows. Array shape: (H*W).
    """
    shape = p.shape[1:]
    dims = len(p)

    pflows = []
    edges = []

    for i in range(dims):
        pflows.append(np.int32(p[i]).flatten())
        edges.append(np.arange(-0.5 - rpad, shape[i] + 0.5 + rpad, 1))  # bins

    h, _ = np.histogramdd(
        tuple(pflows), bins=edges
    )  # (cell seeds) as histogram bins (H, W)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)  # enlarge the seeds

    # non-zero pixel yx-coords of the bins as seeds
    seeds: Tuple[np.ndarray, np.ndarray] = np.nonzero(
        np.logical_and(h - hmax > -1e-6, h > 10)
    )

    Nmax: np.ndarray = h[seeds]  # bincounts, len == n cells
    isort: np.ndarray = np.argsort(Nmax)[::-1]  # sorted bincount indices, descending
    seeds = [s[isort] for s in seeds]  # sort yx-coords by bincount, descending

    return seeds, h, pflows


def expand_seed_pixels(
    seeds: Tuple[np.ndarray, np.ndarray], h: np.ndarray, dims: int = 2
) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
    """Expand the seed pixels to 3x3 neihgborhood of pixels for every seed.

    Parameters
    ----------
        seeds : Tuple[np.ndarray, np.ndarray]
            The y- and x- seeds sorted by descending bincount.
            np.ndarray.shape: (n_cells, ). Dtype: int32.
        h : np.ndarray
            The 2D histogram of the flows. Shape (H+pad, W+pad).
        dims : int, default=2
            The number of dimensions in the input data.

    Returns
    -------
        Tuple[Tuple[np.ndarray, np.ndarray]]:
            The expanded pixel neighborhood coords for all the seeds
    """
    pix: List[Tuple[np.ndarray, np.ndarray]] = list(np.array(seeds).T)
    shape: Tuple[int, int] = h.shape
    expand: Tuple[np.ndarray, np.ndarray] = np.nonzero(np.ones((3, 3)))

    # loop the seed coords
    for k in range(len(pix)):
        # convert tuple to list
        pix[k] = list(pix[k])

        newpix = []
        iin = []
        for i, e in enumerate(expand):
            # centered 1d window around pixel x- or y-coords
            # `pix[k]` can be a  varying length
            epix: np.ndarray = (
                e[:, None] + np.expand_dims(pix[k][i], 0) - 1
            )  # (9, n_pixels)
            epix: np.ndarray = epix.flatten()

            # make sure window is inside the image
            iin.append(np.logical_and(epix >= 0, epix < shape[i]))
            newpix.append(epix)

        # filter pixls whose y and x coords inside img
        iin = np.all(tuple(iin), axis=0)
        newpix = [pi[iin] for pi in newpix]

        # include all pixels with more than 2 final pixels p of the newpix window
        newpix = tuple(newpix)
        igood = h[newpix] > 2
        for i in range(dims):
            pix[k][i] = newpix[i][igood]

        # convert back to tuple
        pix[k] = tuple(pix[k])

    return pix


def get_masks_cellpose(p: np.ndarray, rpad: int = 20) -> np.ndarray:
    """Create masks using pixel convergence after running dynamics.

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------
        p : np.ndarray
            Final locations of each pixel after dynamics. Shape (2, H, W).
        rpad : int, default=20
            Histogram edge padding.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W).
    """
    shape0 = p.shape[1:]
    dims = len(p)

    seeds, h, pflows = get_seeds(p, rpad, dims)
    pix = expand_seed_pixels(seeds, h, dims)

    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad

    # remove big masks
    M0 = M[tuple(pflows)]
    _, counts = np.unique(M0, return_counts=True)
    big = float(np.prod(shape0))
    for i in np.nonzero(counts > big)[0]:
        M0[M0 == i] = 0

    _, M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    return M0


def post_proc_cellpose(
    inst_map: np.ndarray,
    flow_map: np.ndarray,
    return_flows: bool = False,
    min_size: int = 30,
    interp: bool = True,
    use_gpu: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run the cellpose post-processing pipeline.

    https://www.nature.com/articles/s41592-020-01018-x

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
        interp : bool, default=True
            Use bilinear interpolation when integrating the flow dynamics.
        use_gpu : bool, default=True
            Use gpu accelerated bilinear interpolation. If `interp` == False, this is
            ignored.

    Returns
    -------
        np.ndarray:
            The instance labelled segmentation mask. Shape (H, W)
    """
    binary_mask = binarize(inst_map).astype(bool)

    dP = flow_map * binary_mask  # /5.
    # dP = normalize_field(dP)

    pixel_loc, _ = follow_flows(
        dP,
        niter=200,
        mask=binary_mask,
        suppress_euler=False,
        interp=interp,
        use_gpu=use_gpu,
    )

    mask = get_masks_cellpose(pixel_loc)
    inst_map = fill_holes_and_remove_small_masks(mask, min_size=min_size).astype("i4")

    if return_flows:
        hsv_flows = gen_flows(dP)
        return inst_map, hsv_flows

    return inst_map
