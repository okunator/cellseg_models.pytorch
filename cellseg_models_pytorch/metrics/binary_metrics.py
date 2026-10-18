from typing import Tuple

import numpy as np

__all__ = ["get_stats", "iou_score", "dice_coef", "accuracy", "f1score"]


def get_stats(
    true: np.ndarray, pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mark true-positive, false-positive and false-negative pixels.

    Args:
        true: HW binary NumPy mask with integer or boolean values 0 and 1.
        pred: Same-shaped binary prediction mask. Do not supply instance IDs,
            class IDs above 1 or floating probabilities.

    Returns:
        (tp, fp, fn) per-pixel indicator arrays of the input shape. Each marks
        the relevant pixels with 1 (or True), and others with 0. They are not
        already-summed counts; pass them to the binary score functions.

    Example:
        >>> true = np.array([[0, 1, 1]], dtype=np.uint8)
        >>> pred = np.array([[0, 1, 0]], dtype=np.uint8)
        >>> [int(x.sum()) for x in get_stats(true, pred)]
        [1, 0, 1]
    """
    tp = true * pred
    fp = np.bitwise_xor(pred, tp)
    fn = np.bitwise_xor(true, tp)

    return tp, fp, fn


def iou_score(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute binary intersection over union from pixel statistics.

    The score sums the supplied indicator arrays before computing the ratio.

    Args:
        tp: True-positive indicator array from get_stats.
        fp: False-positive indicator array of the same shape.
        fn: False-negative indicator array of the same shape.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Scalar intersection over union, using TP / (TP + FP + FN) with the stated epsilon.
        All-background pairs score zero for overlap/F1 and nearly one for accuracy.
        No thresholding, instance matching or dataset aggregation is performed.
    """
    numerator = tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def dice_coef(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute binary Dice overlap from pixel statistics.

    The score sums the supplied indicator arrays before computing the ratio.

    Args:
        tp: True-positive indicator array from get_stats.
        fp: False-positive indicator array of the same shape.
        fn: False-negative indicator array of the same shape.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Scalar Dice overlap, using 2 * TP / (2 * TP + FP + FN) with the stated epsilon.
        All-background pairs score zero for overlap/F1 and nearly one for accuracy.
        No thresholding, instance matching or dataset aggregation is performed.
    """
    numerator = 2 * tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def accuracy(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute binary pixel accuracy from pixel statistics.

    Accuracy derives true negatives from per-pixel 0/1 indicators, so do not
    supply aggregate counts.

    Args:
        tp: True-positive indicator array from get_stats.
        fp: False-positive indicator array of the same shape.
        fn: False-negative indicator array of the same shape.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Scalar pixel accuracy, using (TP + TN) / number_of_pixels with the stated epsilon.
        All-background pairs score zero for overlap/F1 and nearly one for accuracy.
        No thresholding, instance matching or dataset aggregation is performed.
    """
    tn = 1 - (tp + fn + fp)
    numerator = tn.sum() + tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def f1score(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8) -> float:
    """Compute binary F1 score from pixel statistics.

    The score sums the supplied indicator arrays before computing the ratio.

    Args:
        tp: True-positive indicator array from get_stats.
        fp: False-positive indicator array of the same shape.
        fn: False-negative indicator array of the same shape.
        eps: Small value added to the denominator to avoid division by zero.

    Returns:
        Scalar F1 score, using TP / (TP + 0.5 * FP + 0.5 * FN) with the stated epsilon.
        All-background pairs score zero for overlap/F1 and nearly one for accuracy.
        No thresholding, instance matching or dataset aggregation is performed.
    """
    numerator = tp.sum()
    denominator = 0.5 * fp.sum() + 0.5 * fn.sum() + numerator + eps

    return numerator / denominator
