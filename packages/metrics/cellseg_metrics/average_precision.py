"""Confidence-ranked instance average precision, independent of model outputs."""

from collections.abc import Mapping, Sequence
from math import isfinite

import numpy as np

__all__ = ["precision_recall_ap", "mean_average_precision"]


def precision_recall_ap(
    true: np.ndarray | Sequence[np.ndarray],
    pred: np.ndarray | Sequence[np.ndarray],
    scores: Mapping[int, float] | Sequence[Mapping[int, float]],
    iou_threshold: float = 0.5,
) -> float:
    """Integrate the interpolated precision–recall curve for instance detections.

    Args:
        true: Integer HW reference instance mask, or a sequence of masks. Zero
            is background; positive IDs identify nuclei and need not be contiguous.
        pred: Same-shaped prediction mask, or a corresponding sequence. Images
            may have different sizes. Each pixel belongs to at most one instance.
        scores: Mapping from every nonzero prediction ID to a finite confidence
            score, or one mapping per image. Higher scores mean more confident
            detections; probabilities are not required. IDs may repeat across images.
        iou_threshold: Minimum accepted instance IoU in (0, 1], inclusive.

    Returns:
        AP in [0, 1], using the area under the all-points precision envelope.
        Predictions match the highest-IoU unmatched reference in confidence order.
        Dataset inputs are ranked together, not averaged per image. Equal scores
        retain image order, then ascending prediction ID. Zero with no detections;
        NaN with no reference objects in the entire input.

    Raises:
        ValueError: If masks, score IDs/values, image counts or threshold are invalid.

    Examples:
        >>> true = np.array([[0, 7, 7, 0]])
        >>> pred = np.array([[0, 3, 3, 9]])
        >>> precision_recall_ap(true, pred, {3: 0.8, 9: 0.9})
        0.5
    """
    thresholds = _validate_thresholds([iou_threshold])
    return float(_ap_scores(true, pred, scores, thresholds, recall_samples=None)[0])


def mean_average_precision(
    true: np.ndarray | Sequence[np.ndarray],
    pred: np.ndarray | Sequence[np.ndarray],
    scores: Mapping[int, float] | Sequence[Mapping[int, float]],
    iou_thresholds: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Average confidence-ranked AP over IoU thresholds using COCO-style sampling.

    Args:
        true: Integer HW reference instance mask, or a sequence of masks. Zero
            is background; positive IDs identify objects.
        pred: Same-shaped prediction mask, or a corresponding sequence. Sparse
            IDs are accepted and may repeat across images.
        scores: Finite confidence for every nonzero prediction ID, keyed by ID;
            supply one mapping per image for dataset inputs.
        iou_thresholds: Accepted minimum IoUs in (0, 1], inclusive. None uses
            0.50, 0.55, ..., 0.95. The sequence must be nonempty and contain no duplicates.

    Returns:
        Mean AP in [0, 1]. Each threshold uses the precision envelope sampled at
        101 recall levels from 0 to 1, rather than the exact all-points integral
        in precision_recall_ap. Matching and score ties follow that function.
        Returns zero with no predictions and NaN with no reference objects.
        Evaluates one category using all supplied detections: no COCO crowd/ignore,
        area bins, category averaging or 100-detection limit is applied.

    Raises:
        ValueError: If masks, scores, image counts or thresholds are invalid.

    Examples:
        >>> mask = np.array([[0, 7, 7]])
        >>> mean_average_precision(mask, mask, {7: 0.9})
        1.0
    """
    thresholds = _validate_thresholds(
        np.linspace(0.5, 0.95, 10) if iou_thresholds is None else iou_thresholds
    )
    return float(
        _ap_scores(true, pred, scores, thresholds, np.linspace(0, 1, 101)).mean()
    )


def _validate_thresholds(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate IoU cutoffs without changing their requested order."""
    thresholds = np.asarray(values, dtype=np.float64)
    if (
        thresholds.ndim != 1
        or thresholds.size == 0
        or not np.isfinite(thresholds).all()
        or (thresholds <= 0).any()
        or (thresholds > 1).any()
        or np.unique(thresholds).size != thresholds.size
    ):
        raise ValueError("IoU thresholds must be unique finite values in (0, 1]")
    return thresholds


def _image_matches(
    true: np.ndarray,
    pred: np.ndarray,
    scores: Mapping[int, float],
    thresholds: np.ndarray,
) -> tuple[int, list[int | float], np.ndarray]:
    """Return reference count, ranked confidences and per-threshold detection hits."""
    if not isinstance(true, np.ndarray) or not isinstance(pred, np.ndarray):
        raise ValueError("Instance masks must be NumPy arrays")
    if true.ndim != 2 or pred.shape != true.shape:
        raise ValueError("Instance masks must have matching HW shapes")
    if true.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("Instance masks must contain integer labels")
    if (true < 0).any() or (pred < 0).any():
        raise ValueError("Instance labels must be nonnegative")
    if not isinstance(scores, Mapping):
        raise ValueError("Scores must map prediction IDs to confidences")

    true_ids, true_index, true_area = np.unique(
        true, return_inverse=True, return_counts=True
    )
    pred_ids, pred_index, pred_area = np.unique(
        pred, return_inverse=True, return_counts=True
    )
    foreground_true, foreground_pred = true_ids != 0, pred_ids != 0
    prediction_ids = pred_ids[foreground_pred]
    if set(scores) != set(prediction_ids):
        raise ValueError("Score keys must equal the nonzero prediction IDs")
    # Native numbers keep large integer scores distinct, including across images.
    confidence: list[int | float] = []
    for label in prediction_ids:
        value = scores[label]
        if isinstance(value, np.generic):
            value = value.item()
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and not isfinite(value))
        ):
            raise ValueError("Confidence scores must be finite real numbers")
        confidence.append(value)
    order = sorted(range(len(confidence)), key=confidence.__getitem__, reverse=True)
    ranked = [confidence[index] for index in order]
    count = int(foreground_true.sum())
    hits = np.zeros((len(thresholds), len(prediction_ids)), dtype=bool)
    if count == 0 or len(prediction_ids) == 0:
        return count, ranked, hits

    # Count overlaps once; do not keep a full-image mask for every nucleus.
    intersections = np.bincount(
        true_index.ravel() * len(pred_ids) + pred_index.ravel(),
        minlength=len(true_ids) * len(pred_ids),
    ).reshape(len(true_ids), len(pred_ids))[np.ix_(foreground_true, foreground_pred)]
    union = (
        true_area[foreground_true, None]
        + pred_area[None, foreground_pred]
        - intersections
    )
    iou = intersections / union
    for row, threshold in enumerate(thresholds):
        used = np.zeros(count, dtype=bool)
        for rank, column in enumerate(order):
            overlaps = iou[:, column].copy()
            overlaps[used] = -1
            # Resolve equal overlaps consistently, favoring the larger reference ID.
            match = count - 1 - int(overlaps[::-1].argmax())
            if overlaps[match] >= threshold:
                hits[row, rank] = True
                used[match] = True
    return count, ranked, hits


def _ap_scores(
    true: np.ndarray | Sequence[np.ndarray],
    pred: np.ndarray | Sequence[np.ndarray],
    scores: Mapping[int, float] | Sequence[Mapping[int, float]],
    thresholds: np.ndarray,
    recall_samples: np.ndarray | None,
) -> np.ndarray:
    """Combine per-image matches into globally ranked AP values, one per IoU."""
    if isinstance(true, np.ndarray):
        if not isinstance(pred, np.ndarray) or not isinstance(scores, Mapping):
            raise ValueError(
                "Supply one mask and score mapping, or sequences of all three"
            )
        true, pred, scores = [true], [pred], [scores]
    if isinstance(pred, np.ndarray) or isinstance(scores, Mapping):
        raise ValueError("Supply one mask and score mapping, or sequences of all three")
    if not len(true) == len(pred) == len(scores):
        raise ValueError("Reference, prediction and score image counts must match")

    reference_count = 0
    confidences: list[int | float] = []
    matches: list[np.ndarray] = []
    for reference, prediction, confidence in zip(true, pred, scores):
        count, ranked, hits = _image_matches(
            reference, prediction, confidence, thresholds
        )
        reference_count += count
        confidences.extend(ranked)
        matches.append(hits)
    if reference_count == 0:
        return np.full(len(thresholds), np.nan)

    order = sorted(range(len(confidences)), key=confidences.__getitem__, reverse=True)
    hits = np.concatenate(matches, axis=1)[:, order]
    cumulative = hits.cumsum(axis=1, dtype=np.float64)
    recall = cumulative / reference_count
    precision = cumulative / np.arange(1, hits.shape[1] + 1)
    # Later, more complete detections can improve the precision envelope at earlier recall.
    precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    if recall_samples is None:
        increments = np.diff(recall, axis=1, prepend=0)
        return (increments * precision).sum(axis=1)

    sampled = np.zeros((len(thresholds), len(recall_samples)))
    for row in range(len(thresholds)):
        positions = np.searchsorted(recall[row], recall_samples, side="left")
        valid = positions < hits.shape[1]
        sampled[row, valid] = precision[row, positions[valid]]
    return sampled.mean(axis=1)
