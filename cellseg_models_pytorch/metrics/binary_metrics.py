from typing import Tuple

import numpy as np

__all__ = ["get_stats", "iou_score", "dice_coef", "accuracy", "f1score"]


def get_stats(
    true: np.ndarray, pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute basic metrics for diagnostic tests.

    Args:
        true: Ground truth binary mask. Shape (H, W)
        pred: Predicted binary mask. Shape (H, W)

    Returns:
        The computed TP, FP, and FN pixels in a labelled mask.
        Shapes: (H, W)
    """
    tp = true * pred
    fp = np.bitwise_xor(pred, tp)
    fn = np.bitwise_xor(true, tp)

    return tp, fp, fn


def iou_score(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute the intersection over union (Jaccard-index).

    Args:
        tp: True positive pixels. Shape (H, W).
        fp: False postive pixels. Shape (H, W).
        fn: False negative pixels. Shape (H, W).
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        The computed intersection over union.
    """
    numerator = tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def dice_coef(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute the Sørensen-Dice coefficient.

    Args:
        tp: True positive pixels. Shape (H, W).
        fp: False postive pixels. Shape (H, W).
        fn: False negative pixels. Shape (H, W).
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        The computed dice coefficient.
    """
    numerator = 2 * tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def accuracy(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8
) -> float:
    """Compute the binary accuracy.

    Args:
        tp: True positive pixels. Shape (H, W).
        fp: False postive pixels. Shape (H, W).
        fn: False negative pixels. Shape (H, W).
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        The computed binary accuracy.
    """
    tn = 1 - (tp + fn + fp)
    numerator = tn.sum() + tp.sum()
    denominator = fp.sum() + fn.sum() + numerator + eps

    return numerator / denominator


def f1score(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the binary f1-score.

    Args:
        tp: True positive pixels. Shape (H, W).
        fp: False postive pixels. Shape (H, W).
        fn: False negative pixels. Shape (H, W).
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        The computed f1-score.
    """
    numerator = tp.sum()
    denominator = 0.5 * fp.sum() + 0.5 * fn.sum() + numerator + eps

    return numerator / denominator
