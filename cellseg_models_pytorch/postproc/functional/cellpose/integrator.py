"""
Imported from the cellpose repo.

https://github.com/MouseLand/cellpose/blob/master/cellpose/dynamics.py

Removed unnecessary stuff + code
style mods + docstrings for readability.

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
from typing import Tuple

import numpy as np
import torch
from numba import njit, prange

__all__ = ["steps2D_interp", "steps2D", "follow_flows"]


def steps2D_interp(
    p: np.ndarray,
    dP: np.ndarray,
    niter: int = 200,
    suppress_euler: bool = False,
    use_gpu: bool = True,
) -> np.ndarray:
    """Run bilinear interpolation on non-zero pixel locations.

    NOTE: uses torch.nn.functional.grid_sample.

    Run Euler integration.

    "run a dynamical system starting at that pixel location and
     following the spatial derivatives specified by the horizontal and
     vertical gradient maps. We use finite differences with a step size
     of one."

    Parameters
    ----------
        p : np.ndarray
            All the pixel locations for the pixels in the flow maps.
            Including zero-pixels. Shape: (2, H, W). Dtype: float32
        dP : np.ndarray
            Flow maps. Shape: (2, H, W). Dtype: float64
        niter : int, default=200
            Number of iterations of dynamics to run
        suppress_euler : bool, default=False
            Suppress euler step. Used for omnipose.
        use_gpu : bool, default = True
            Flag, wheter to use gpu.

    Returns
    -------
        np.ndarray:
            The pixel locations after dynamics. Shape (2, H, W).

    """
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    shape = dP.shape[1:]
    shape = np.array(shape)[[1, 0]].astype("double") - 1

    # Flip dims and modify to shape [1, 1, H*W, 2] for `grid_sample`
    pt = torch.from_numpy(p[[1, 0]].T).double().to(device)
    pt = pt.unsqueeze(0).unsqueeze(0)

    im = torch.from_numpy(dP[[1, 0]]).double().to(device)
    im = im.unsqueeze(0)

    # normalize pt between 0 and 1, normalize the flow
    for k in range(2):
        im[:, k, ...] *= 2.0 / shape[k]
        pt[..., k] /= shape[k]

    # normalize to between -1 and 1
    pt = pt * 2 - 1

    # Euler integration steps
    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)

        if suppress_euler:
            dPt /= 1 + t

        # clamp the final pixel locations
        for k in range(2):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k, ...], -1.0, 1.0)

    # Undo the normalization from before, reverse order of operations
    pt = (pt + 1) * 0.5
    for k in range(2):
        pt[..., k] *= shape[k]

    return pt[..., [1, 0]].cpu().numpy().squeeze().T


@njit(parallel=True)
def steps2D(
    p: np.ndarray,
    dP: np.ndarray,
    inds: np.ndarray,
    niter: int = 200,
    suppress_euler: bool = False,
) -> np.ndarray:
    """Run dynamics of pixels to recover masks in 2D.

    Euler integration of dynamics dP for niter steps

    Parameters
    ----------
        p : np.ndarray
            All the pixel locations for the pixels in the flow maps.
            Including zero-pixels. Shape: (2, H, W). Dtype: float32
        dP : np.ndarray
            Flow maps. Shape: (2, H, W). Dtype: float64
        inds: np.ndarray
            Non-zero pixels to run dynamics on. Shape: (npixels, 2). Dtype: int32.
        niter : int, default=200
            Number of iterations of dynamics to run.
        suppress_euler : bool, default=False
            Suppress euler step. Used for omnipose.

    Returns
    -------
        np.ndarray:
            Final locations of each pixel after dynamics. Shape (2, H, W).
            Dtype: float32.
    """
    shape = p.shape[1:]
    for t in prange(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]

            if suppress_euler:
                step /= 1 + t

            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))

    return p


def follow_flows(
    dP: np.ndarray,
    mask: np.ndarray = None,
    niter: int = 200,
    suppress_euler: bool = False,
    interp: bool = True,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Define pixels and run dynamics to recover masks in 2D.

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds).

    Parameters
    ----------
        dP : np.ndarray
            Flow maps. Shape: (2, H, W). Dtype: float64.
        mask : np.ndarray, default=None
            Pixel mask to seed masks. Useful when flows have low magnitudes.
        niter : int, default=200
            Number of iterations of dynamics to run.
        suppress_euler : bool, default=False
            Suppression factor for the euler intergator. Used for omnipose.
        interp : bool, default=True
            Use bilinear interpolation when integrating.
        use_gpu : bool, default=True
            Use gpu accelerated bilinear interpolation. If `interp` == False, this is
            ignored.

    Returns
    ---------------
        Tuple[np.ndarray, np.ndarray]:
            A tuple of nd.arrays. The first index is teh final locations
            of each pixel after dynamics (Shape: (2, H, W)) The second
            index is the indices of the non-zero pixels.
            Shape: (number of non zero pixels, 2)
    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    # run dynamics on subset of pixels.
    # NOTE: Uses only xmap or ymap (dP[0|1])
    seeds = np.abs(dP[0]) > 1e-3
    if mask is not None:
        seeds = np.logical_or(mask, seeds)

    pixel_loc = np.nonzero(seeds)
    inds = np.array(pixel_loc).astype(np.int32).T

    # Sometimes a random error in empty images.. errr... dunno...
    if interp:
        try:
            p[:, inds[:, 0], inds[:, 1]] = steps2D_interp(
                p=p[:, inds[:, 0], inds[:, 1]],
                dP=dP,
                niter=niter,
                suppress_euler=suppress_euler,
                use_gpu=use_gpu,
            )
        except Exception:
            pass
    else:
        try:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
        except Exception:
            pass

    return p, inds
