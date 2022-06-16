"""
Adapted from the cellpose repo.

https://github.com/MouseLand/cellpose

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
import scipy.ndimage as ndi
from numba import njit

__all__ = ["normalize_field", "gen_flow_maps"]


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


@njit(nogil=True)
def _extend_centers(
    T: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    ymed: int,
    xmed: int,
    w: int,
    niter: int,
) -> np.ndarray:
    """Run diffusion from center of the label."""
    for _ in range(niter):
        T[ymed * w + xmed] += 1

        t = T[y * w + x]
        tdy = T[(y - 1) * w + x] + T[(y + 1) * w + x]
        tdx = T[y * w + x - 1] + T[y * w + x + 1]
        tdydx = T[(y - 1) * w + x - 1] + T[(y - 1) * w + x + 1]
        tdxdy = T[(y + 1) * w + x - 1] + T[(y + 1) * w + x + 1]

        T[y * w + x] = 1 / 9.0 * (t + tdy + tdx + tdydx + tdxdy)

    return T


def gen_flow_maps(inst_map: np.ndarray, pad: int = 1) -> np.ndarray:
    """Generate flow maps from inst maps like in CellPose.

    I.e. take the x- and y- derivates from a time-independent heat
    diffusion map of the instances.

    Stringer, C., Wang, T., Michaelos, M. et al. Cellpose:
    a generalist algorithm for cellular segmentation. Nat Methods 18,
    100-106 (2021). https://doi.org/10.1038/s41592-020-01018-x

    https://www.nature.com/articles/s41592-020-01018-x

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        pad : int, default=1
            Number of pixels for constant padding.

    Returns
    -------
        np.ndarray:
            Y and X- flows in this order. Shape (2, H, W). Dtype: float64
    """
    H, W = inst_map.shape
    mu = np.zeros((2, H, W), np.float64)
    mu_c = np.zeros((H, W), np.float64)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    # loop the instances
    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.int32)

        # padded bounding box of the instance
        sy, sx = ndi.find_objects(inst)[0]
        inst = np.pad(inst[sy, sx], pad)
        h, w = inst.shape

        # non-zero pixel indices in the bounding box
        y, x = np.nonzero(inst)

        # number of iterations for heat diffusion equation
        niter = 2 * np.int32((np.ptp(x) + np.ptp(y)))

        # center of mass for the instance
        ymed = np.median(y)
        xmed = np.median(x)
        imin = np.argmin((x - xmed) ** 2 + (y - ymed) ** 2)
        xmed = np.array([x[imin]], np.int32)
        ymed = np.array([y[imin]], np.int32)

        # solve heat equation
        T = np.zeros(h * w, dtype=np.float64)
        T = _extend_centers(T, y, x, ymed, xmed, np.int32(w), np.int32(niter))
        T[(y + 1) * w + x + 1] = np.log(1.0 + T[(y + 1) * w + x + 1])

        # central difference approximation to first derivative
        dy = (T[(y + 1) * w + x] - T[(y - 1) * w + x]) / 2
        dx = (T[y * w + x + 1] - T[y * w + x - 1]) / 2

        mu[:, (sy.start + y - pad), (sx.start + x - pad)] = np.stack([dy, dx])
        mu_c[sy.start + y - pad, sx.start + x - pad] = T[y * w + x]  # heat diff

    mu = normalize_field(mu)

    return mu
