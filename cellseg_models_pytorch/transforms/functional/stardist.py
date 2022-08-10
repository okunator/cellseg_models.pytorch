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
from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
from numba import njit
from skimage.measure import find_contours, regionprops
from skimage.morphology import dilation, disk

from ...utils import remove_small_objects

__all__ = ["evenly_spaced_points", "gen_radial_distmaps", "gen_stardist_maps"]


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


def evenly_spaced_points(
    binary_map: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find evenly spaced points on a contour of an object.

    Interpolates evenly spaced points from the arc length of the contour.

    Based on:
    https://stackoverflow.com/questions/27429784/equally-spaced-points-in-a-contour

    Parameters
    ----------
        binary_map : np.ndarray
            A binary mask containing one object/nuclei.
        n : int
            Number of evenly spaced points

    Returns
    -------
        Tuple[np.ndarray, np.ndarray]:
            The y- and x- coordinates of the evenly spaced points. Shapes: (n*).
    """
    n_objs = len(np.unique(binary_map))
    if n_objs > 2:
        raise ValueError(
            f"Expects the input to contain only one object. Got: {n_objs}."
        )

    contours = find_contours(binary_map)[0]
    xc = contours[:, 1]
    yc = contours[:, 0]

    # spacing of x and y points.
    dy = np.diff(yc)
    dx = np.diff(xc)

    # distances between consecutive coordinates
    dS = np.sqrt(dx**2 + dy**2)
    dS = np.append(np.zeros(1), dS)  # include starting point

    # Arc length and perimeter
    d = np.cumsum(dS)
    perim = d[-1]

    # divide the perimeter to evenly spaced values
    ds = perim / n
    dSi = np.arange(0, n) * ds
    dSi[-1] = dSi[-1] - 5e-3

    # interpolate the x and y coordinates
    yi = np.interp(dSi, d, yc)
    xi = np.interp(dSi, d, xc)

    return yi, xi


# TODO: njit this
def gen_radial_distmaps(
    inst_map: np.ndarray, n_rays: int = 32, n_segments: int = 15
) -> np.ndarray:
    """Compute radial distances for each non-zero pixel.

    This is a python (workaround) implementation of the stardist
    radial distance computation algorithm. Results in smoother distance
    maps than the original.

    NOTE: This implementation is still slower than the C implementation
    found in the stardist library. However, a lot faster than the python
    implementation found in the same library. With default params atleast
    x16 faster

    Here, for every object, evenly spaced points are computed from the
    contour of the object and those points are used as line segments.
    The distance for every pixel is computed from the intersection point
    of the rays and line segments rather than shooting rays from each
    pixel.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        n_rays : int, default=32
            Number of rays.
        n_segments : int, default=15
            Number of line segments the contour is divided into.
            The more segments used, the more detail is preserved with
            performance tradeoff.

    Returns
    -------
        np.ndarray:
            The radial distance maps. Shape (n_rays, H, W). Dtype: float64
    """
    eps = 1e-5
    H, W = inst_map.shape
    dist = np.zeros((n_rays, H, W), np.float32)

    inst_map = remove_small_objects(inst_map, min_size=5, out=inst_map)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.int32)

        # major axis length of the object for padding
        props = regionprops(inst)
        pad = int(props[0].major_axis_length)

        sy, sx = ndi.find_objects(inst)[0]
        inst = np.pad(inst[sy, sx], pad)
        h, w = inst.shape
        y, x = np.nonzero(inst)

        # Evenly spaced point along the contour of the object.
        # The consecutive points are used as line segments
        yi, xi = evenly_spaced_points(dilation(inst, disk(1)), n_segments)
        yj = list(yi[1:])
        xj = list(xi[1:])
        yj.append(yi[0])
        xj.append(xi[0])
        points1 = np.array([yi, xi]).T
        points2 = np.array([yj, xj]).T

        ray_origins = np.array([y, x]).T
        st_rays = np.float32((2 * np.pi) / n_rays)
        for k in range(n_rays):
            theta = k * st_rays + 1e-4  # straight/right angles behave oddly (+ 1e-4)
            ray_directions = np.array([np.sin(theta), np.cos(theta)]).T

            # rays for each pixel with angle theta
            rays = (pad * 1.5) * ray_directions + ray_origins

            # Compute intersecting point for each contour line segment and each ray

            # ray endpoints
            y1 = ray_origins[:, 0]
            x1 = ray_origins[:, 1]
            y2 = rays[:, 0]
            x2 = rays[:, 1]

            dst = np.zeros((n_segments, h * w), dtype=np.float32)
            for t, (p1, p2) in enumerate(zip(points1, points2)):
                # line segments end points
                y3 = p1[0]
                x3 = p1[1]
                y4 = p2[0]
                x4 = p2[1]

                # find intersections
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 1e-10
                det1 = x1 * y2 - y1 * x2
                det2 = x3 * y4 - y3 * x4

                px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
                py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom

                # Set non-intersecting points to NaN
                px[(px - x1) * (px - x2) > eps] = np.nan
                px[(px - x3) * (px - x4) > eps] = np.nan
                py[(py - y1) * (py - y2) > eps] = np.nan
                py[(py - y3) * (py - y4) > eps] = np.nan

                d = ((y - py)) ** 2 + ((x - px)) ** 2
                d = np.sqrt(d) - 1.1  # crude correction for crossing boundaries

                dst[t, y * w + x] = d

            ray_dists = np.nansum(dst, axis=0)
            dist[k, sy.start + y - pad, sx.start + x - pad] = ray_dists[y * w + x]

    return dist
