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
import torch

from .cellpose import normalize_field

__all__ = ["smooth_distance", "gen_omni_flow_maps"]


def smooth_distance(
    inst_map: np.ndarray, device: str = "cpu", pad: int = 1
) -> np.ndarray:
    """Smooth FIM (Fast Iterative Method) based distance transform.

    Avoids boundary pixellation that can cause artefacts in flows.

    https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        device str : default="cpu"
            One of "cuda" or "cpu".
        pad : int, default=1
            The number of pixels padded around the input.

    Returns
    -------
        np.ndarray:
            Smooth distance transform. Shape: (H, W).
    """
    dists = ndi.distance_transform_edt(inst_map)
    inst_map = np.pad(inst_map, pad)
    H, W = inst_map.shape

    # get non-zero pixel indices
    y, x = np.nonzero(inst_map)

    # 9-pixel neighborhood for every nonzero pixel
    yneighbors = np.stack((y - 1, y - 1, y - 1, y, y, y, y + 1, y + 1, y + 1), axis=0)
    xneighbors = np.stack((x - 1, x, x + 1, x - 1, x, x + 1, x - 1, x, x + 1), axis=0)

    # get neighbor validator (not all neighbors are in same mask)
    # extract list of label values,
    neighbor_masks = inst_map[yneighbors, xneighbors]
    isneighbor = neighbor_masks == neighbor_masks[4]  # 4 corresponds to x, y now

    # set number of iterations
    n_iter = np.ceil(np.max(dists) * 1.16).astype(int) + 1
    nimg = xneighbors.shape[0] // 9

    # solve the eikonal heat eq
    pty = torch.from_numpy(yneighbors).to(device)
    ptx = torch.from_numpy(xneighbors).to(device)
    T = torch.zeros((nimg, H, W), dtype=torch.double, device=device)
    isneigh = torch.from_numpy(isneighbor).to(device)

    for _ in range(n_iter):
        Tneigh = T[:, pty, ptx]
        Tneigh *= isneigh

        # using flattened index for the lattice points, just like gradient below
        minx = torch.minimum(Tneigh[:, 3, :], Tneigh[:, 5, :])
        mina = torch.minimum(Tneigh[:, 2, :], Tneigh[:, 6, :])
        miny = torch.minimum(Tneigh[:, 1, :], Tneigh[:, 7, :])
        minb = torch.minimum(Tneigh[:, 0, :], Tneigh[:, 8, :])

        A = torch.where(
            torch.abs(mina - minb) >= 2,
            torch.minimum(mina, minb) + np.sqrt(2),
            0.5 * (mina + minb + torch.sqrt(4 - (mina - minb) ** 2)),
        )
        B = torch.where(
            torch.abs(miny - minx) >= np.sqrt(2),
            torch.minimum(miny, minx) + 1,
            0.5 * (miny + minx + torch.sqrt(2 - (miny - minx) ** 2)),
        )

        T[:, pty[4, ...], ptx[4, ...]] = torch.sqrt(A * B + 1e-12)

    return T.cpu().squeeze().numpy()[pad:-pad, pad:-pad]


# Adapted from
# https://github.com/MouseLand/cellpose/blob/master/cellpose/omnipose/core.py
def gen_omni_flow_maps(inst_map: np.ndarray, pad: int = 1) -> np.ndarray:
    """Generate eikonal flow maps from inst maps like in OmniPose.

    I.e. take the first order x- and -y derivatives of eikonal distance
    transfroms of the cell masks.

    Omnipose: a high-precision morphology-independent solution for
    bacterial cell segmentation Kevin J. Cutler, Carsen Stringer,
    Paul A. Wiggins, Joseph D. Mougous bioRxiv 2021.11.03.467199;
    doi: https://doi.org/10.1101/2021.11.03.467199

    https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        pad (int, default=1):
            number of pixels for constant padding

    Returns
    -------
        np.ndarray:
            Y and X- flows in this order. Shape (2, H, W). Dtype: float64.
    """
    # FMI euclidean distance transform
    dists = smooth_distance(inst_map)

    # number of iterations for heat diffusion equation
    niter = np.ceil(np.max(dists) * 1.16).astype(int) + 1

    H, W = inst_map.shape
    mu = np.zeros((2, H, W), np.float64)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.int32)

        # padded bounding box of the instance
        sy, sx = ndi.find_objects(inst)[0]
        inst = np.pad(inst[sy, sx], pad)
        h, w = inst.shape

        # non-zero pixel indices in the bounding box
        y, x = np.nonzero(inst)

        # solve heat equation for eikonal distance
        T = np.zeros(h * w, dtype=np.float64)
        for _ in range(niter):
            minx = np.minimum(T[y * w + x - 1], T[y * w + x + 1])
            miny = np.minimum(
                T[(y - 1) * w + x],
                T[(y + 1) * w + x],
            )
            mina = np.minimum(T[(y - 1) * w + x - 1], T[(y + 1) * w + x + 1])
            minb = np.minimum(T[(y - 1) * w + x + 1], T[(y + 1) * w + x - 1])

            a = 4 - (mina - minb) ** 2
            a[a < 0] = 0.0
            A = np.where(
                np.abs(mina - minb) >= 2,
                np.minimum(mina, minb) + np.sqrt(2),
                0.5 * (mina + minb + np.sqrt(a)),
            )

            b = 2 - (miny - minx) ** 2
            b[b < 0] = 0.0
            B = np.where(
                np.abs(miny - minx) >= np.sqrt(2),
                np.minimum(miny, minx) + 1,
                0.5 * (miny + minx + np.sqrt(b)),
            )

            T[y * w + x] = np.sqrt(A * B + 1e-12)

        # central difference approximation to first derivative
        dy = (T[(y + 1) * w + x] - T[(y - 1) * w + x]) / 2
        dx = (T[y * w + x + 1] - T[y * w + x - 1]) / 2

        mu[:, (sy.start + y - pad), (sx.start + x - pad)] = np.stack([dy, dx])

    mu = normalize_field(mu)

    return mu
