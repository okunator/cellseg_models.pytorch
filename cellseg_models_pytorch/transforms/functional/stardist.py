"""
Adapted `gen_stardist_maps` form the stardist repo.

- https://github.com/stardist/stardist

BSD 3-Clause License

Copyright (c) 2018-2022, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from numba import njit

__all__ = ["gen_stardist_maps"]


@njit
def gen_stardist_maps(inst_map: np.ndarray, n_rays: int) -> np.ndarray:
    """Compute radial distances for each non-zero pixel in a label mask.

    NOTE: Adapted from
    - https://github.com/stardist/stardist/blob/master/stardist/geometry/geom2d.py

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        n_rays : int, default=32
            Number of rays.

    Returns
    -------
        np.ndarray:
            The radial distance maps. Shape (n_rays, H, W). Dtype: float32
    """
    n_rays = int(n_rays)
    dist = np.empty(inst_map.shape + (n_rays,), np.float32)

    st_rays = np.float32((2 * np.pi) / n_rays)
    for i in range(inst_map.shape[0]):
        for j in range(inst_map.shape[1]):
            value = inst_map[i, j]
            if value == 0:
                dist[i, j] = 0
            else:
                for k in range(n_rays):
                    phi = np.float32(k * st_rays)
                    dy = np.cos(phi)
                    dx = np.sin(phi)
                    x, y = np.float32(0), np.float32(0)
                    while True:
                        x += dx
                        y += dy
                        ii = int(round(i + x))
                        jj = int(round(j + y))
                        if (
                            ii < 0
                            or ii >= inst_map.shape[0]
                            or jj < 0
                            or jj >= inst_map.shape[1]
                            or value != inst_map[ii, jj]
                        ):
                            # small correction as we overshoot the boundary
                            t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                            x -= t_corr * dx
                            y -= t_corr * dy
                            dst = np.sqrt(x**2 + y**2)
                            dist[i, j, k] = dst
                            break

    return dist.transpose(2, 0, 1)
