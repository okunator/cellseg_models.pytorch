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

from numbers import Integral

import numpy as np
import torch
from numba import njit, prange

__all__ = ["steps2D_interp", "steps2D", "follow_flows", "follow_flows_torch"]


def _validate_flow_map(dP: torch.Tensor, niter: int) -> None:
    if dP.ndim != 3 or dP.shape[0] != 2 or min(dP.shape[1:]) < 1:
        raise ValueError("Flows must have shape (2, H, W) with nonempty spatial axes.")
    if dP.dtype not in (torch.float32, torch.float64):
        raise TypeError("Flows must use float32 or float64.")
    if dP.device.type not in ("cpu", "cuda"):
        raise ValueError("Flow integration supports CPU and CUDA; float64 is required.")
    if isinstance(niter, bool) or not isinstance(niter, Integral) or niter < 0:
        raise ValueError("niter must be a nonnegative integer.")
    if not torch.isfinite(dP).all():
        raise ValueError("Flows must contain only finite values.")


@torch.no_grad()
def _steps2D_interp_torch(
    p: torch.Tensor,
    dP: torch.Tensor,
    niter: int,
    suppress_euler: bool,
) -> torch.Tensor:
    """Integrate validated (2, N) pixel coordinates in float64 on their device."""
    if p.shape[1] == 0 or niter == 0:
        return p.to(dtype=torch.float64).clone()
    extent = (dP.shape[2] - 1, dP.shape[1] - 1)
    pt = p.flip(0).T.to(dtype=torch.float64)[None, None]
    im = dP.flip(0).to(dtype=torch.float64)[None]
    for k, size in enumerate(extent):
        if size:
            im[:, k] *= 2.0 / size
            pt[..., k] /= size
        else:
            # A singleton axis samples its center and cannot move.
            im[:, k] = 0
            pt[..., k] = 0.5
    pt = pt * 2 - 1
    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        if suppress_euler:
            dPt /= 1 + t
        for k in range(2):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1.0, 1.0)
    pt = (pt + 1) * 0.5
    for k, size in enumerate(extent):
        pt[..., k] *= size
    return pt[0, 0].flip(-1).T


@torch.no_grad()
def follow_flows_torch(
    dP: torch.Tensor,
    mask: torch.Tensor | None = None,
    niter: int = 200,
    suppress_euler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow CellPose/Omnipose flows without copying tensors to another device.

    Args:
        dP: Finite float32 or float64 (2, H, W) flow values, in (y, x) channel
            order, on CPU or CUDA. These are displacements, not probabilities.
            Pass the masked CellPose flows or divergence-rescaled Omnipose flows.
        mask: Optional bool (H, W) seed mask on the same device. Seeds are its
            union with pixels whose absolute x-flow exceeds 1e-3, preserving
            the NumPy integrator's selection rule.
        niter: Number of bilinear Euler integration steps, including zero.
        suppress_euler: Divide each step by its one-based iteration for Omnipose.

    Returns:
        Float32 (2, H, W) final (y, x) pixel coordinates and int64 (N, 2) seed
        indices, both on the input device. Unseeded pixels keep their coordinates.
        Integration uses float64, zero-padded sampling with align_corners=False,
        and image-border clamping. Inputs are unchanged; outputs have no autograd
        history. Instance labels are produced by a separate reconstruction stage.

    Raises:
        TypeError: Flows are not float32/float64, or mask is not bool.
        ValueError: Shapes/devices differ, values are nonfinite, iterations are
            invalid, or the device cannot run this CPU/CUDA float64 path.

    Note:
        Finite-value validation and dynamic seed selection can synchronize CUDA.
        No NumPy conversion or dense tensor transfer is performed. MPS does not
        support this float64 path; use CPU tensors on a Mac.

    Example:
        >>> import torch
        >>> from cellseg_models_pytorch.postproc import follow_flows_torch
        >>> flows = torch.zeros((2, 5, 7), dtype=torch.float32, device="cpu")
        >>> endpoints, indices = follow_flows_torch(flows, niter=4)
        >>> endpoints.shape, indices.shape
        (torch.Size([2, 5, 7]), torch.Size([0, 2]))
    """
    _validate_flow_map(dP, niter)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("The seed mask must use bool.")
        if mask.shape != dP.shape[1:] or mask.device != dP.device:
            raise ValueError("The seed mask must match the flow shape and device.")
    height, width = dP.shape[1:]
    p = torch.stack(
        torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=dP.device),
            torch.arange(width, dtype=torch.float32, device=dP.device),
            indexing="ij",
        )
    )
    seeds = dP[1].abs() > 1e-3
    if mask is not None:
        seeds = seeds | mask
    inds = torch.nonzero(seeds)
    y, x = inds.unbind(1)
    p[:, y, x] = _steps2D_interp_torch(p[:, y, x], dP, niter, suppress_euler).to(
        dtype=p.dtype
    )
    return p, inds


def steps2D_interp(
    p: np.ndarray,
    dP: np.ndarray,
    niter: int = 200,
    suppress_euler: bool = False,
    use_gpu: bool = True,
) -> np.ndarray:
    """Integrate selected NumPy pixel coordinates with bilinear Euler steps.

    Args:
        p: Finite float32/float64 (2, N) starting coordinates in (y, x) order.
        dP: Finite float32/float64 (2, H, W) flows in (y, x) order.
        niter: Nonnegative number of integration steps.
        suppress_euler: Divide each step by its one-based iteration for Omnipose.
        use_gpu: Use the current CUDA device when available; otherwise use CPU.

    Returns:
        Float64 NumPy coordinates of shape (2, N), including N=0 or N=1.
        Inputs are unchanged. Float64 sampling and image-border clamping match
        ``follow_flows_torch``. Use that function to keep tensor inputs on-device.

    Raises:
        TypeError: Flows or coordinates do not use float32/float64.
        ValueError: Shapes, values or iteration counts are invalid.
    """
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    flows = torch.from_numpy(np.ascontiguousarray(dP)).to(device)
    points = torch.from_numpy(np.ascontiguousarray(p)).to(device)
    _validate_flow_map(flows, niter)
    if points.ndim != 2 or points.shape[0] != 2:
        raise ValueError("Coordinates must have shape (2, N).")
    if points.dtype not in (torch.float32, torch.float64):
        raise TypeError("Coordinates must use float32 or float64.")
    if not torch.isfinite(points).all():
        raise ValueError("Coordinates must contain only finite values.")
    return _steps2D_interp_torch(points, flows, niter, suppress_euler).cpu().numpy()


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

    Args:
        p: All the pixel locations for the pixels in the flow maps.
            Including zero-pixels. Shape: (2, H, W). Dtype: float32
        dP: Flow maps. Shape: (2, H, W). Dtype: float64
        inds:
            Non-zero pixels to run dynamics on. Shape: (npixels, 2). Dtype: int32.
        niter: Number of iterations of dynamics to run.
            Defaults to 200.
        suppress_euler: Suppress euler step. Used for omnipose.
            Defaults to False.

    Returns:
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
    mask: np.ndarray | None = None,
    niter: int = 200,
    suppress_euler: bool = False,
    interp: bool = True,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow a NumPy flow field and return pixel endpoints and seed indices.

    Args:
        dP: Float32/float64 (2, H, W) flows in (y, x) order. Bilinear
            integration requires finite values and nonempty spatial axes.
        mask: Optional (H, W) seed mask. Nonzero values add seed pixels to
            those whose absolute x-flow exceeds 1e-3.
        niter: Nonnegative number of integration steps.
        suppress_euler: Divide bilinear steps by their one-based iteration
            for Omnipose.
        interp: Use bilinear tensor integration. False retains the existing
            nearest-neighbor Numba implementation.
        use_gpu: Use the current CUDA device for bilinear integration when
            available; otherwise use CPU. Ignored when interp is False.

    Returns:
        Float32 NumPy (2, H, W) final (y, x) pixel coordinates and int32 (N, 2)
        seed indices. Unseeded pixels keep their coordinates. Bilinear
        integration uses float64 internally and leaves inputs unchanged.

    Raises:
        TypeError: Bilinear flows do not use float32/float64.
        ValueError: Bilinear inputs or iteration counts are invalid.

    Note:
        Bilinear integration errors propagate to the caller. Use
        ``follow_flows_torch`` to avoid the NumPy/device transfer boundary.
    """
    if interp:
        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        flows = torch.from_numpy(np.ascontiguousarray(dP)).to(device)
        seeds = (
            torch.from_numpy(np.ascontiguousarray(mask)).to(
                device=device, dtype=torch.bool
            )
            if mask is not None
            else None
        )
        p, inds = follow_flows_torch(flows, seeds, niter, suppress_euler)
        return p.cpu().numpy(), inds.cpu().numpy().astype(np.int32)

    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    # run dynamics on subset of pixels.
    # NOTE: Uses only xmap or ymap (dP[0|1])
    seeds = np.abs(dP[1]) > 1e-3
    if mask is not None:
        seeds = np.logical_or(mask, seeds)

    pixel_loc = np.nonzero(seeds)
    inds = np.array(pixel_loc).astype(np.int32).T

    try:
        p = steps2D(p, dP.astype(np.float32), inds, niter)
    except Exception:
        pass

    return p, inds
