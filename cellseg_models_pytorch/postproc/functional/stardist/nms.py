import math
from typing import List, Sequence, Tuple

import numpy as np
from numba import njit, prange
from scipy.spatial import KDTree

from cellseg_models_pytorch.utils import intersection

__all__ = ["get_bboxes", "nms_stardist"]


@njit(parallel=False)
def get_bboxes(
    dist: np.ndarray, points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Get bounding boxes from the non-zero pixels of the radial distance maps.

    This is basically a translation from the stardist repo cpp code to python

    NOTE: jit compiled and parallelized with numba.

    Parameters
    ----------
        dist : np.ndarray
            The non-zero values of the radial distance maps. Shape: (n_nonzero, n_rays).
        points : np.ndarray
            The yx-coordinates of the non-zero points. Shape (n_nonzero, 2).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        Returns the x0, y0, x1, y1 bbox coordinates, bbox areas and the maximum
        radial distance in the image.
    """
    n_polys = dist.shape[0]
    n_rays = dist.shape[1]

    bbox_x1 = np.zeros(n_polys)
    bbox_x2 = np.zeros(n_polys)
    bbox_y1 = np.zeros(n_polys)
    bbox_y2 = np.zeros(n_polys)

    areas = np.zeros(n_polys)
    angle_pi = 2 * math.pi / n_rays
    max_dist = 0

    for i in prange(n_polys):
        max_radius_outer = 0
        py = points[i, 0]
        px = points[i, 1]

        for k in range(n_rays):
            d = dist[i, k]
            y = py + d * np.sin(angle_pi * k)
            x = px + d * np.cos(angle_pi * k)

            if k == 0:
                bbox_x1[i] = x
                bbox_x2[i] = x
                bbox_y1[i] = y
                bbox_y2[i] = y
            else:
                bbox_x1[i] = min(x, bbox_x1[i])
                bbox_x2[i] = max(x, bbox_x2[i])
                bbox_y1[i] = min(y, bbox_y1[i])
                bbox_y2[i] = max(y, bbox_y2[i])

            max_radius_outer = max(d, max_radius_outer)

        areas[i] = (bbox_x2[i] - bbox_x1[i]) * (bbox_y2[i] - bbox_y1[i])
        max_dist = max(max_dist, max_radius_outer)

    return bbox_x1, bbox_y1, bbox_x2, bbox_y2, areas, max_dist


@njit
def _suppress_bbox(
    query: Sequence[int],
    current_idx: int,
    boxes: np.ndarray,
    areas: np.ndarray,
    suppressed: List[bool],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Inner loop of the stardist nms algorithm where bboxes are suppressed.

    NOTE: Numba compiled only for performance.
          Parallelization had only a negative effect on run-time on.
          12-core hyperthreaded Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz.
    """
    for i in range(len(query)):
        query_idx = query[i]

        if suppressed[query_idx]:
            continue

        overlap = intersection(boxes[current_idx], boxes[query_idx])
        iou = overlap / min(areas[current_idx] + 1e-10, areas[query_idx] + 1e-10)
        suppressed[query_idx] = iou > iou_threshold

    return suppressed


def nms_stardist(
    boxes: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    areas: np.ndarray,
    max_dist: float,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Non maximum suppression for stardist bboxes.

    NOTE: This implementation relies on `scipy.spatial` `KDTree`

    NOTE: This version of nms is faster than the original one in stardist repo
    and is fully written in python. The differenecs in the resulting instance
    segmentation masks are neglible.

    Parameters
    ----------
        boxes : np.ndarray
            An array of bbox coords in pascal VOC format (x0, y0, x1, y1).
            Shape: (n_points, 4). Dtype: float64.
        points : np.ndarray
            The yx-coordinates of the non-zero points. Shape (n_points, 2). Dtype: int64
        scores : np.ndarray
            The probability values at the point coordinates. Shape (n_points,).
            Dtype: float32/float64.
        areas : np.ndarray
            The areas of the bounding boxes at the point coordinates. Shape (n_points,).
            Dtype: float32/float64.
        radius_outer : np.ndarray
            The radial distances to background at each point. Shape (n_points, )
        max_dist : float
            The maximum radial distance of all the radial distances
        score_threshold : float, default=0.5
            Threshold for the probability distance map.
        iou_threshold : float, default=0.5
            Threshold for the IoU metric deciding whether to suppres a bbox.

    Returns
    -------
        np.ndarray:
            The indices of the bboxes that are not suppressed. Shape: (n_kept, ).
    """
    keep = []

    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)

    kdtree = KDTree(points, leafsize=16)

    suppressed = np.full(len(boxes), False)
    for current_idx in range(len(scores)):
        # If already visited or discarded
        if suppressed[current_idx]:
            continue

        # If score is already below threshold then break
        if scores[current_idx] < score_threshold:
            break

        # Query the points
        query = kdtree.query_ball_point(points[current_idx], max_dist)
        suppressed = _suppress_bbox(
            np.array(query), current_idx, boxes, areas, suppressed, iou_threshold
        )

        # Add the current box
        keep.append(current_idx)

    return np.array(keep)
