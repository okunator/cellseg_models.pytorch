"""Backward-compatible imports from the standalone metrics package."""

from cellseg_metrics.functional import (
    pairwise_pixel_stats as pairwise_pixel_stats,
    pairwise_object_stats as pairwise_object_stats,
    panoptic_quality as panoptic_quality,
    average_precision as average_precision,
    aggregated_jaccard_index as aggregated_jaccard_index,
    dice2 as dice2,
    iou_multiclass as iou_multiclass,
    dice_multiclass as dice_multiclass,
    f1score_multiclass as f1score_multiclass,
    accuracy_multiclass as accuracy_multiclass,
    sensitivity_multiclass as sensitivity_multiclass,
    specificity_multiclass as specificity_multiclass,
    _match_instances as _match_instances,
    _absent_inds as _absent_inds,
    get_stats as get_stats,
    iou_score as iou_score,
)

__all__ = [
    "pairwise_pixel_stats",
    "pairwise_object_stats",
    "panoptic_quality",
    "average_precision",
    "aggregated_jaccard_index",
    "dice2",
    "iou_multiclass",
    "dice_multiclass",
    "f1score_multiclass",
    "accuracy_multiclass",
    "sensitivity_multiclass",
    "specificity_multiclass",
]
