"""Copied the polygons to label utilities from stardist repo (with minor modifications).

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
from skimage.draw import polygon
from skimage.morphology import disk, erosion

from .nms import get_bboxes, nms_stardist

__all__ = ["post_proc_stardist_orig", "polygons_to_label"]


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

    ind = np.argsort(prob, kind="stable")
    points = points[ind]
    dist = dist[ind]

    coord = dist_to_coord(dist, points, scale_dist=scale_dist)

    return polygons_to_label_coord(coord, shape=shape, labels=ind)


def post_proc_stardist(
    dist_map: np.ndarray,
    stardist_map: np.ndarray,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.4,
    trim_bboxes: bool = True,
    normalize: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run post-processing for stardist outputs.

    NOTE: This is not the original cpp version.
    This is a python re-implementation of the stardidst post-processing
    pipeline that uses non-maximum-suppression. Here, critical parts of the
    nms are accelerated with `numba` and `scipy.spatial.KDtree`.

    NOTE:
    This implementaiton of the stardist post-processing is actually nearly 2x
    faster than the original version if `trim_bboxes` is set to True. The resulting
    segmentation is not an exact match but the differences are mostly neglible.

    Parameters
    ----------
        dist_map : np.ndarray
            Predicted distance transform. Shape: (H, W).
        stardist_map : np.ndarray
            Predicted radial distances. Shape: (n_rays, H, W).
        score_thresh : float, default=0.4
            Threshold for the regressed distance transform.
        iou_thresh : float, default=0.4
            Threshold for the non-maximum suppression.
        trim_bboxes : bool, default=True
            If True, The non-zero pixels are computed only from the cell contours
            which prunes down the pixel search space drastically.
        normalize : bool, default=True
            If True, the distance transform is normalized to the range 0-1.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape: (H, W).
    """
    if (
        not dist_map.ndim == 2
        and not stardist_map.ndim == 3
        and not dist_map.shape == stardist_map.shape[:2]
    ):
        raise ValueError(
            "Illegal input shapes. Make sure that: "
            f"`dist_map` has to have shape: (H, W). Got: {dist_map.shape} "
            f"`stardist_map` has to have shape (n_rays, H, W). Got: {stardist_map.shape}"
        )
    shape = dist_map.shape
    dist = np.asarray(stardist_map).transpose(1, 2, 0).astype(np.float32)
    prob = np.asarray(dist_map).astype(np.float32)

    # normalize prob array if values are not in range 0-1
    if normalize:
        prob = (prob - prob.min()) / (prob.max() - prob.min())

    # threshold the edt distance transform map
    mask = _ind_prob_thresh(prob, score_thresh)

    # get only the mask contours to trim down bbox search space
    if trim_bboxes:
        fp = disk(2)
        mask -= erosion(mask, fp)

    points = np.stack(np.where(mask), axis=1)

    # Get only non-zero pixels of the transforms
    dist = dist[mask > 0]
    scores = prob[mask > 0]

    # sort descendingly
    ind = np.argsort(scores)[::-1]
    dist = dist[ind]
    scores = scores[ind]
    points = points[ind]

    # get bounding boxes
    x1, y1, x2, y2, areas, max_dist = get_bboxes(dist, points)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # consider only boxes above score threshold
    score_cond = scores >= score_thresh
    boxes = boxes[score_cond]
    scores = scores[score_cond]
    areas = areas[score_cond]

    # run nms
    inds = nms_stardist(
        boxes,
        points,
        scores,
        areas,
        max_dist,
        score_threshold=score_thresh,
        iou_threshold=iou_thresh,
    )

    # get the points, scores, and dists of the boxes that are above the score
    # threshold and are not suppressed by nms
    points = points[inds]
    scores = scores[inds]
    dist = dist[inds]
    labels = polygons_to_label(dist, points, prob=scores, shape=shape)

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
    dist_map = dist_map.squeeze()
    if dist_map.max() > 1 or dist_map.min() < 0:
        dist_map = (dist_map - dist_map.min()) / (dist_map.max() - dist_map.min())

    img_shape = dist_map.shape
    stardist_map = stardist_map.transpose(1, 2, 0)
    points, probi, disti = non_maximum_suppression(
        stardist_map, dist_map.squeeze(), prob_thresh=thresh
    )
    labels = polygons_to_label(
        disti, points, prob=probi, shape=img_shape, scale_dist=rescale
    )

    return labels
