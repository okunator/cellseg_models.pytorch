"""Ported from:
https://github.com/jopo666/HistoPrep/tree/master/histoprep/functional/_dearray.py

MIT License

Copyright (c) 2023 jopo666

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import warnings

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def get_spot_coordinates(
    spot_mask: np.ndarray,
) -> dict[str, tuple[int, int, int, int]]:
    """Dearray tissue microarray spots based on a spot mask.

    Numbers each spot from top-left and takes into account missing spots etc in the
    numbering.

    Parameters:
        spot_mask (np.ndarray):
            Tissue mask of TMA-slide, should only contain TMA spots an no artifacts.

    Returns:
        dict[str, tuple[int, int, int, int]]:
            Dictionary of spot numbers and xywh-coordinates.
    """
    # Detect contours and get their bboxes and centroids.
    bboxes, centroids = _contour_bboxes_and_centroids(spot_mask)
    if len(bboxes) == 0:
        return {}
    # Detect possible rotation of the image based on centroids.
    centroids = _rotate_coordinates(centroids, _detect_rotation(centroids))
    # Detect optimal number of rows and columns and cluster each spot.
    num_cols = _optimal_cluster_size(centroids[:, 0].reshape(-1, 1))
    num_rows = _optimal_cluster_size(centroids[:, 1].reshape(-1, 1))
    col_labels = _hierachial_clustering(
        centroids[:, 0].reshape(-1, 1), n_clusters=num_cols
    )
    row_labels = _hierachial_clustering(
        centroids[:, 1].reshape(-1, 1), n_clusters=num_rows
    )
    # Change label numbers to correct order (starting from top-left).
    x_means = [centroids[col_labels == i, 0].mean() for i in range(num_cols)]
    y_means = [centroids[row_labels == i, 1].mean() for i in range(num_rows)]
    for i in range(num_cols):
        new_label = np.arange(num_cols)[np.argsort(x_means) == i]
        col_labels[col_labels == i] = -new_label
    col_labels *= -1
    for i in range(num_rows):
        new_label = np.arange(num_rows)[np.argsort(y_means) == i]
        row_labels[row_labels == i] = -new_label
    row_labels *= -1
    # Collect numbers.
    numbers = np.zeros(len(centroids)).astype("str")
    current_number = 1
    same_spot_number = False
    for r in range(num_rows):
        for c in range(num_cols):
            matches = [x == (c, r) for x in zip(col_labels, row_labels)]
            if sum(matches) == 1:
                numbers[matches] = str(current_number)
            elif sum(matches) > 1:
                same_spot_number = True
                numbers[matches] = [
                    f"{current_number}-{version + 1}" for version in range(sum(matches))
                ]
            current_number += 1
    if same_spot_number:
        warnings.warn("Some spots were assigned the same number.", stacklevel=1)
    # Return bboxes and numbers.
    return {f"spot_{k}": tuple(v) for k, v in zip(numbers, bboxes)}


def _contour_bboxes_and_centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract contour bounding boxes and centroids."""
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return np.array([]), np.array([])
    bboxes, centroids = [], []
    for cnt, is_parent in zip(contours, hierarchy[0][:, -1] == -1):
        # Skip non-parents and contours without area.
        if not is_parent or cv2.contourArea(cnt) == 0:
            continue
        # Get bounding box.
        bboxes.append(cv2.boundingRect(cnt))
        # Get centroid.
        moments = cv2.moments(cnt)
        centroids.append(
            (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        )
    return np.array(bboxes), np.array(centroids)


def _detect_rotation(centroids: np.ndarray) -> float:
    """Detect rotation from centroid coordinates and return angle in radians."""
    # Calculate angle between each centroid.
    n = len(centroids)
    thetas = []
    for r in range(n):
        for c in range(n):
            x1, y1 = centroids[r]
            x2, y2 = centroids[c]
            thetas.append(np.rad2deg(np.arctan2(y2 - y1, x2 - x1)))
    # We want deviations from 0 so divide corrections.
    corrections = np.arange(0, 361, 45)
    for i, theta in enumerate(thetas):
        sign = np.sign(theta)
        idx = np.abs(np.abs(theta) - corrections).argmin()
        thetas[i] = theta - sign * corrections[idx]
    # Finally return most common angle.
    values, counts = np.unique(np.round(thetas), return_counts=True)
    theta = values[counts.argmax()]
    return np.radians(theta)


def _rotate_coordinates(coords: np.ndarray, theta: float) -> np.ndarray:
    """Rotate coordinates with given theta."""
    c, s = np.cos(theta), np.sin(theta)
    r_matrix = np.array(((c, -s), (s, c)))
    return coords @ r_matrix


def _optimal_cluster_size(data: np.ndarray) -> int:
    """Find optimal cluster size for dataset X."""
    sil = []
    if data.shape[0] <= 2:  # noqa
        return 1
    for n in range(2, data.shape[0]):
        labels = _hierachial_clustering(data=data, n_clusters=n)
        sil.append(silhouette_score(data, labels))
    return np.argmax(sil) + 2


def _hierachial_clustering(data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Perform hierarchian clustering and get labels."""
    if n_clusters == 1:
        return np.zeros(data.shape[0], dtype=np.int32)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clust.fit(data)
    return clust.labels_
