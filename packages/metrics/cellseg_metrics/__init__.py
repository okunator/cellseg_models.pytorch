"""NumPy/SciPy metrics for binary, multiclass and instance segmentation."""

__version__ = "0.1.0"

from .binary_metrics import accuracy, dice_coef, f1score, get_stats, iou_score
from .functional import (
    accuracy_multiclass,
    aggregated_jaccard_index,
    average_precision,
    dice2,
    dice_multiclass,
    f1score_multiclass,
    iou_multiclass,
    mean_average_precision,
    pairwise_object_stats,
    pairwise_pixel_stats,
    panoptic_quality,
    precision_recall_ap,
    sensitivity_multiclass,
    specificity_multiclass,
)

__all__ = [
    "get_stats",
    "accuracy",
    "iou_score",
    "f1score",
    "dice_coef",
    "pairwise_pixel_stats",
    "panoptic_quality",
    "dice2",
    "average_precision",
    "precision_recall_ap",
    "mean_average_precision",
    "pairwise_object_stats",
    "aggregated_jaccard_index",
    "iou_multiclass",
    "dice_multiclass",
    "f1score_multiclass",
    "accuracy_multiclass",
    "sensitivity_multiclass",
    "specificity_multiclass",
]
