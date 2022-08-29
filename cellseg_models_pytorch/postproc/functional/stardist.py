"""Imported most of the stuff from stardist repo. Minor modifications.

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
from skimage import img_as_ubyte
from skimage.draw import polygon
from skimage.measure import regionprops

from ...utils import bounding_box, remap_label, remove_small_objects
from .drfns import find_maxima, h_minima_reconstruction

__all__ = ["post_proc_stardist", "post_proc_stardist_orig", "polygons_to_label"]


def polygons_to_label_coord(
    coord: np.ndarray, shape: Tuple[int, int], labels: np.ndarray = None
) -> np.ndarray:
    """Render polygons to image given a shape.

    Parameters
    ----------
        coord.shape : np.ndarray
            Shape: (n_polys, n_rays)
        shape : Tuple[int, int]
            Shape of the output mask.
        labels : np.ndarray, optional
            Sorted indices of the centroids.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape: (H, W).
    """
    coord = np.asarray(coord)
    if labels is None:
        labels = np.arange(len(coord))

    assert coord.ndim == 3 and coord.shape[1] == 2 and len(coord) == len(labels)

    lbl = np.zeros(shape, np.int32)

    for i, c in zip(labels, coord):
        rr, cc = polygon(*c, shape)
        lbl[rr, cc] = i + 1

    return lbl


def ray_angles(n_rays: int = 32):
    """Get linearly spaced angles for rays."""
    return np.linspace(0, 2 * np.pi, n_rays, endpoint=False)


def dist_to_coord(
    dist: np.ndarray, points: np.ndarray, scale_dist: Tuple[int, int] = (1, 1)
) -> np.ndarray:
    """Convert list of distances and centroids from polar to cartesian coordinates.

    Parameters
    ----------
        dist : np.ndarray
            The centerpoint pixels of the radial distance map. Shape (n_polys, n_rays).
        points : np.ndarray
            The centroids of the instances. Shape: (n_polys, 2).
        scale_dist : Tuple[int, int], default=(1, 1)
            Scaling factor.

    Returns
    -------
        np.ndarray:
            Cartesian cooridnates of the polygons. Shape (n_polys, 2, n_rays).
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    assert (
        dist.ndim == 2
        and points.ndim == 2
        and len(dist) == len(points)
        and points.shape[1] == 2
        and len(scale_dist) == 2
    )
    n_rays = dist.shape[1]
    phis = ray_angles(n_rays)
    coord = (dist[:, np.newaxis] * np.array([np.sin(phis), np.cos(phis)])).astype(
        np.float32
    )
    coord *= np.asarray(scale_dist).reshape(1, 2, 1)
    coord += points[..., np.newaxis]
    return coord


def _ind_prob_thresh(prob: np.ndarray, prob_thresh: float, b: int = 2) -> np.ndarray:
    """Index based thresholding."""
    if b is not None and np.isscalar(b):
        b = ((b, b),) * prob.ndim

    ind_thresh = prob > prob_thresh
    if b is not None:
        _ind_thresh = np.zeros_like(ind_thresh)
        ss = tuple(
            slice(_bs[0] if _bs[0] > 0 else None, -_bs[1] if _bs[1] > 0 else None)
            for _bs in b
        )
        _ind_thresh[ss] = True
        ind_thresh &= _ind_thresh
    return ind_thresh.astype("int32")


def polygons_to_label(
    dist: np.ndarray,
    points: np.ndarray,
    shape: Tuple[int, int],
    prob: np.ndarray = None,
    thresh: float = -np.inf,
    scale_dist: Tuple[int, int] = (1, 1),
) -> np.ndarray:
    """Convert distances and center points to instance labelled mask.

    Parameters
    ----------
        dist : np.ndarray
            The centerpoint pixels of the radial distance map. Shape (n_polys, n_rays).
        points : np.ndarray
            The centroids of the instances. Shape: (n_polys, 2).
        shape : Tuple[int, int]:
            Shape of the output mask.
        prob : np.ndarray, optional
            The centerpoint pixels of the regressed distance transform.
            Shape: (n_polys, n_rays).
        thresh : float, default=-np.inf
            Threshold for the regressed distance transform.
        scale_dist : Tuple[int, int], default=(1, 1)
            Scaling factor.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W).
    """
    dist = np.asarray(dist)
    points = np.asarray(points)
    prob = np.inf * np.ones(len(points)) if prob is None else np.asarray(prob)

    assert dist.ndim == 2 and points.ndim == 2 and len(dist) == len(points)
    assert len(points) == len(prob) and points.shape[1] == 2 and prob.ndim == 1

    ind = prob > thresh
    points = points[ind]
    dist = dist[ind]
    prob = prob[ind]

    ind = np.argsort(prob, kind="stable")
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def _clean_up(inst_map: np.ndarray, size: int = 150, **kwargs) -> np.ndarray:
    """Clean up overlapping instances."""
    mask = remap_label(inst_map.copy())
    mask_connected = ndi.label(mask)[0]

    labels_connected = np.unique(mask_connected)[1:]
    for lab in labels_connected:
        inst = np.array(mask_connected == lab, copy=True)
        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= mask_connected.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= mask_connected.shape[0] - 1 else y2

        box_insts = mask[y1:y2, x1:x2]
        if len(np.unique(ndi.label(box_insts)[0])) <= 2:
            real_labels, counts = np.unique(box_insts, return_counts=True)
            real_labels = real_labels[1:]
            counts = counts[1:]
            max_pixels = np.max(counts)
            max_label = real_labels[np.argmax(counts)]
            for real_lab, count in list(zip(list(real_labels), list(counts))):
                if count < max_pixels:
                    if count < size:
                        mask[mask == real_lab] = max_label

    return mask


def post_proc_stardist(
    dist_map: np.ndarray, stardist_map: np.ndarray, thresh: float = 0.4, **kwargs
) -> np.ndarray:
    """Run post-processing for stardist.

    NOTE: This is not the original version that uses NMS.
    This is rather a workaround that is a little slower.

    Parameters
    ----------
        dist_map : np.ndarray
            Predicted distance transform. Shape: (H, W).
        stardist_map : np.ndarray
            Predicted radial distances. Shape: (n_rays, H, W).
        thresh : float, default=0.4
            Threshold for the regressed distance transform.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape: (H, W).
    """
    stardist_map = stardist_map.transpose(1, 2, 0)
    mask = _ind_prob_thresh(dist_map, thresh, b=2)

    # invert distmap
    inv_dist_map = 255 - img_as_ubyte(dist_map)

    # find markers from minima erosion reconstructed maxima of inv dist map
    reconstructed = h_minima_reconstruction(inv_dist_map)
    markers = find_maxima(reconstructed, mask=mask)
    markers = ndi.label(markers)[0]
    markers = remove_small_objects(markers, min_size=5)
    points = np.array(
        tuple(np.array(r.centroid).astype(int) for r in regionprops(markers))
    )

    if len(points) == 0:
        return np.zeros_like(mask)

    dist = stardist_map[tuple(points.T)]
    scores = dist_map[tuple(points.T)]

    labels = polygons_to_label(
        dist, points, prob=scores, shape=mask.shape, scale_dist=(1, 1)
    )
    labels = _clean_up(labels, **kwargs)

    return labels


def post_proc_stardist_orig(
    dist_map: np.ndarray, stardist_map: np.ndarray, thresh: float = 0.4, **kwargs
) -> np.ndarray:
    """Run the original stardist post-processing pipeline.

    NOTE: to use this the `stardist` package needs to be installed.

    Parameters
    ----------
        dist_map : np.ndarray
            Predicted distance transform. Shape: (H, W).
        stardist_map : np.ndarray
            Predicted radial distances. Shape: (n_rays, H, W).
        thresh : float, default=0.4
            Threshold for the regressed distance transform.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape: (H, W).
    """
    try:
        from stardist import non_maximum_suppression
    except ImportError:
        raise ImportError(
            "Install stardist if using the original post-proc method. "
            "`pip install stardist`"
        )
    rescale = (1, 1)
    img_shape = dist_map.shape
    stardist_map = stardist_map.transpose(1, 2, 0)
    points, probi, disti = non_maximum_suppression(
        stardist_map, dist_map, prob_thresh=thresh
    )
    labels = polygons_to_label(
        disti, points, prob=probi, shape=img_shape, scale_dist=rescale
    )

    return labels
