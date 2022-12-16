"""
Imported from cellpose repo (Old version). Code style mods.

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

from typing import Union

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d

from cellseg_models_pytorch.utils import binarize

from .utils import fill_holes_and_remove_small_masks

__all__ = ["post_proc_cellpose_old"]


def steps2D_interp_torch(p: np.ndarray, dP: np.ndarray, niter: int = 200) -> np.ndarray:
    """Run dynamics.

    "run a dynamical system starting at that pixel location and
     following the spatial derivatives specified by the horizontal and
     vertical gradient maps. We use finite differences with a step size
     of one."

    Parameters
    ----------
        p : float32, 3D array
            final locations of each pixel after dynamics,
            size [axis x Ly x Lx].
        dP : float32, 3D array
            flows [axis x Ly x Lx]
        niter : int (optional, default 200)
            number of iterations of dynamics to run

    Returns
    -------
        np.ndarray:
            The pixel locations after dynamics. Shape (2, H, W).
    """
    shape = dP.shape[1:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pt = torch.from_numpy(p[[1, 0]].T).double().to(device)
    pt = pt.unsqueeze(0).unsqueeze(0)
    pt[..., 0] = pt[..., 0] / (shape[1] - 1)  # normalize to between  0 and 1
    pt[..., 1] = pt[..., 1] / (shape[0] - 1)  # normalize to between  0 and 1
    pt = pt * 2 - 1  # normalize to between -1 and 1
    im = torch.from_numpy(dP[[1, 0]]).double().to(device)
    im = im.unsqueeze(0)

    for k in range(2):
        im[:, k, ...] /= (shape[1 - k] - 1) / 2.0

    for _ in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        for k in range(2):
            pt[..., k] = torch.clamp(pt[..., k] - dPt[:, k, ...], -1.0, 1.0)

    pt = (pt + 1) * 0.5
    pt[..., 0] = pt[..., 0] * (shape[1] - 1)
    pt[..., 1] = pt[..., 1] * (shape[0] - 1)

    return pt[..., [1, 0]].cpu().numpy().squeeze().T


def follow_flows(dP: np.ndarray, niter: int = 200) -> np.ndarray:
    """Define pixels and run dynamics to recover masks in 2D.

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------
        dP : float32, 3D
            flows [axis x Ly x Lx]
        niter : int (optional, default 200)
            number of iterations of dynamics to run

    Returns
    -------
        p : float32, 3D array
            Final locations of each pixel after dynamics
    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.int32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

    p = np.array(p).astype(np.float32)

    # run dynamics on subset of pixels. NOTE: Uses only xmap or ymap (dP[0|1])
    inds = np.array(np.nonzero(np.abs(dP[1]) > 1e-3)).astype(np.int32).T

    # Interpolation mode
    # Sometimes a random error in empty images.. errr... dunno...
    try:
        p[:, inds[:, 0], inds[:, 1]] = steps2D_interp_torch(
            p=p[:, inds[:, 0], inds[:, 1]],
            dP=dP,
            niter=niter,
        )
    except Exception:
        pass

    return p


def get_masks_cellpose_old(
    p: np.ndarray,
    threshold: int = 0.4,
    rpad: int = 20,
    flows: Union[np.ndarray, None] = None,
    iscell: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """Create masks using pixel convergence after running dynamics.

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------
        p: float32, 3D array
            final locations of each pixel after dynamics,
            size [axis x Ly x Lx].
        rpad: int (optional, default 20)
            histogram edge padding
        threshold: float (optional, default 0.4)
            masks with flow error greater than threshold are discarded
            (if flows is not None)
        flows: float, 3D array (optional, default None)
            flows [axis x Ly x Lx]. If flows
            is not None, then masks with inconsistent flows are removed using
            `remove_bad_flow_masks`.
        iscell: bool, 2D array
            if iscell is not None, set pixels that are
            iscell False to stay in their original location.
    Returns
    -------
        M0: int, 2D array
            masks with inconsistent flow masks removed,
            0=NO masks; 1,2,...=mask labels,
            size [Ly x Lx]

    """
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]), indexing="ij")

        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

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


def post_proc_cellpose_old(inst_map: np.ndarray, flow_map: np.ndarray) -> np.ndarray:
    """Run the CellPose post-processing pipeline (OLD).

    https://www.nature.com/articles/s41592-020-01018-x

    Note:
    Does not discard bad flows which are removed in the original paper.
    In general the flows are somewhat useless... except for nice viz..

    Args:
    -----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        flow_map : np.ndarray
            Horizontal and vertical flows. Shape (2, H, W).

    Returns
    -------
        np.ndarray:
            The instance labelled segmentation mask. Shape (H, W)

    """
    binary_mask = binarize(inst_map).astype(bool)
    # dP = -1 * dp * binary_mask / 5.0 # Weird result, Dunno ??
    dP = flow_map * binary_mask
    pixel_loc = follow_flows(dP, niter=3000)

    mask = get_masks_cellpose_old(
        p=pixel_loc, iscell=binary_mask, flows=dP, threshold=None
    )
    inst_map = fill_holes_and_remove_small_masks(mask, min_size=30)

    return inst_map.astype("i4")
