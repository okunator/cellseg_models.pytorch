"""Backward-compatible imports from the standalone metrics package."""

from cellseg_metrics.binary_metrics import (
    get_stats as get_stats,
    iou_score as iou_score,
    dice_coef as dice_coef,
    accuracy as accuracy,
    f1score as f1score,
)

__all__ = ["get_stats", "iou_score", "dice_coef", "accuracy", "f1score"]
