from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .binary_metrics import get_stats, iou_score

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


def pairwise_pixel_stats(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int | None = None,
    metric_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
) -> list[np.ndarray]:
    """Build overlap statistics for ground-truth and prediction label pairs.

    Args:
        true: Integer HW NumPy mask. In instance mode, 0 is background and
            positive IDs identify objects; IDs need not be consecutive.
        pred: Same-shaped integer prediction mask.
        num_classes: None compares sorted nonzero instance IDs. An integer
            compares class IDs 0 through num_classes-1, including class 0.
        metric_func: Optional function accepting tp, fp, fn pixel indicators,
            for example iou_score, to produce one value per overlapping pair.

    Returns:
        List of float64 pairwise matrices. Without metric_func, returns
        [tp, fp, fn] counts; with it, a one-element list containing that score.
        Rows correspond to sorted true IDs and columns to sorted predicted IDs,
        or to all class IDs in class mode. Non-overlapping label pairs retain
        zeros in every matrix, including fp/fn. Instance mode returns an empty
        list if either mask has no foreground; class mode includes absent classes.
    """
    if num_classes is not None:
        true_labels = list(range(num_classes))
        pred_labels = list(range(num_classes))
    else:
        true_labels = list(np.unique(true[true != 0]))
        pred_labels = list(np.unique(pred[pred != 0]))

    true_indices = {label: ix for ix, label in enumerate(true_labels)}
    pred_indices = {label: ix for ix, label in enumerate(pred_labels)}

    true_objects = {}
    for t in true_labels:
        true_obj = np.array(true == t, np.uint8)
        true_objects[t] = true_obj

    pred_objects = {}
    for p in pred_labels:
        pred_obj = np.array(pred == p, np.uint8)
        pred_objects[p] = pred_obj

    # array dims
    i = len(true_labels)
    j = len(pred_labels)

    # init return list
    ret = []
    if i > 0 and j > 0:
        ret.append(np.zeros((i, j), dtype=np.float64))
        if metric_func is None:
            ret.append(np.zeros((i, j), dtype=np.float64))
            ret.append(np.zeros((i, j), dtype=np.float64))

    for true_label in true_labels:
        true_obj = true_objects[true_label]

        overlap = pred[true_obj > 0]
        overlap_label = np.unique(overlap)
        for pred_label in overlap_label:
            # ignore bg and empty preds in instance mode
            if pred_label == 0 and num_classes is None:
                continue

            pred_obj = pred_objects[pred_label]
            tp, fp, fn = get_stats(true_obj, pred_obj)

            ix = true_indices[true_label]
            jx = pred_indices[pred_label]

            # compute a metric or add stats
            if metric_func is not None:
                ret[0][ix, jx] = metric_func(tp, fp, fn)
            else:
                ret[0][ix, jx] = tp.sum()
                ret[1][ix, jx] = fp.sum()
                ret[2][ix, jx] = fn.sum()

    return ret


def pairwise_object_stats(
    matches: np.ndarray, sum_reduce: bool = True
) -> tuple[int, int, int] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count detected, spurious and missed objects from a match matrix.

    Args:
        matches: Boolean matrix (true_objects, predicted_objects). Supply
            one-to-one matches when counting true-positive pairs; this helper
            does not resolve ambiguous assignments.
        sum_reduce: True returns totals. False returns the indicator arrays.

    Returns:
        (tp, fp, fn) integers, or boolean arrays when sum_reduce=False.
        TP and FN arrays index ground-truth rows; FP indexes prediction columns.
        TP counts matched rows rather than every True matrix element.
    """
    true_hits = matches.any(axis=1)
    pred_hits = matches.any(axis=0)

    tp_objects = true_hits
    fp_objects = ~pred_hits
    fn_objects = ~true_hits

    if sum_reduce:
        return int(tp_objects.sum()), int(fp_objects.sum()), int(fn_objects.sum())

    return tp_objects, fp_objects, fn_objects


def _match_instances(iou: np.ndarray, thresh: float) -> np.ndarray:
    """Maximize valid pair count, then summed IoU, with unique row/column matches."""
    matches = iou > thresh
    if thresh < 0.5:
        # One additional valid pair must outweigh every possible IoU tie-break.
        weights = matches * (1 + iou / (min(iou.shape) + 1))
        rows, cols = linear_sum_assignment(weights, maximize=True)
        assigned = np.zeros_like(matches)
        assigned[rows, cols] = matches[rows, cols]
        return assigned
    return matches


def panoptic_quality(
    true: np.ndarray, pred: np.ndarray, thresh: float = 0.5, eps: float = 1e-8
) -> dict[str, float]:
    """Compute panoptic quality with one-to-one instance matching.

    Below 0.5 IoU, maximize the number of qualifying pairs, then their summed
    IoU. Lowering the threshold can increase detection quality while decreasing
    panoptic quality because pair count takes priority over summed IoU.
    At or above 0.5, qualifying pairs are already unique.

    Args:
        true: Integer instance mask (H, W), with zero as background.
        pred: Integer instance mask (H, W), with zero as background.
            Instance IDs need not be contiguous or match the ground truth IDs.
        thresh: IoU must be strictly greater than this value in [0, 1].
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        Dictionary with pq (panoptic quality), sq (mean matched-object overlap)
        and dq (object detection F1). All three are zero if either mask has no
        foreground. The function evaluates one mask pair without cell-type
        classification or dataset aggregation.

    Raises:
        ValueError: If thresh is outside [0, 1] or not finite.

    Examples:
        >>> mask = np.array([[0, 1, 1]])
        >>> round(float(panoptic_quality(mask, mask)["pq"]), 6)
        1.0
    """
    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be finite and between 0 and 1")
    iou = pairwise_pixel_stats(true, pred, metric_func=iou_score)
    res = {"pq": 0.0, "sq": 0.0, "dq": 0.0}

    if iou:
        iou = iou[0]
        matches = _match_instances(iou, thresh)
        tp_objects, fp_objects, fn_objects = pairwise_object_stats(matches)

        dq = tp_objects / (tp_objects + 0.5 * fp_objects + 0.5 * fn_objects + eps)
        sq = iou[matches].sum() / (tp_objects + eps)
        pq = dq * sq

        res["pq"] = pq
        res["sq"] = sq
        res["dq"] = dq

    return res


def average_precision(
    true: np.ndarray, pred: np.ndarray, thresh: float = 0.5, eps: float = 1e-8
) -> float:
    """Compute object precision, TP / (TP + FP), at one IoU threshold.

    Despite the historical name, this is not area under a precision-recall
    curve. Match instances uniquely, maximizing qualifying pair count then IoU.

    Args:
        true: Integer instance mask (H, W), with zero as background.
        pred: Integer instance mask (H, W), with zero as background.
            Instance IDs need not be contiguous or match the ground truth IDs.
        thresh: IoU must be strictly greater than this value in [0, 1].
        eps: Epsilon to avoid zero div errors.
            Defaults to 1e-8.

    Returns:
        Matched prediction fraction at this threshold. Zero if either mask has
        no foreground. This is object precision, not TP / (TP + FP + FN),
        and does not integrate a confidence-ranked precision-recall curve.

    Raises:
        ValueError: If thresh is outside [0, 1] or not finite.

    Examples:
        >>> mask = np.array([[0, 1, 1]])
        >>> round(average_precision(mask, mask), 6)
        1.0
    """
    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be finite and between 0 and 1")
    iou = pairwise_pixel_stats(true, pred, metric_func=iou_score)
    ap = 0.0

    if iou:
        iou = iou[0]
        matches = _match_instances(iou, thresh)
        tp_objects, fp_objects, _ = pairwise_object_stats(matches)

        ap = tp_objects / (tp_objects + fp_objects + eps)

    return ap


def dice2(true: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the package's pairwise instance Dice aggregate.

    Args:
        true: Integer HW instance mask; 0 is background, positive IDs are objects.
        pred: Same-shaped prediction mask. Object IDs need not match true IDs.
        eps: Small denominator offset.

    Returns:
        Scalar from summed pairwise intersections and fp/fn entries for
        overlapping object pairs. This does not use one-to-one object matching;
        objects with no overlapping pair do not contribute to these matrices.
        Returns zero if either mask has no foreground. Do not interpret it as
        ordinary binary Dice on the two foreground masks.
    """
    dice2 = 0.0
    stats = pairwise_pixel_stats(true, pred)

    if stats:
        tp, fp, fn = stats

        numerator = 2 * tp[tp > 0].sum()
        denominator = numerator + fp[fp > 0].sum() + fn[fn > 0].sum() + eps
        dice2 = numerator / denominator

    return dice2


def aggregated_jaccard_index(
    true: np.ndarray, pred: np.ndarray, thresh: float = 0.5, eps: float = 1e-8
) -> float:
    """Aggregate instance overlap using each true object's largest intersection.

    Args:
        true: Integer HW instance mask; 0 is background and positive IDs are objects.
        pred: Same-shaped prediction mask. IDs may be nonconsecutive and unrelated
            to true IDs.
        thresh: Retained argument, currently unused. It does not filter matches.
        eps: Small denominator offset.

    Returns:
        Scalar intersection/union aggregate. Matching uses largest intersection
        area per true object, not largest IoU. Ties can select several predictions
        and the same prediction can be selected for several true objects.
        Unmatched objects contribute their area to the denominator. Returns zero
        if either mask has no foreground. This is not unique AJI+ matching.
    """
    aji = 0.0
    stats = pairwise_pixel_stats(true, pred)

    if stats:
        tp, fp, fn = stats

        inter = tp
        union = tp + fp + fn

        # Get the number of pixels from the matched objects
        matches = inter == np.amax(inter, axis=1, keepdims=True, initial=1e-8)
        inter = inter[matches].sum()
        union = union[matches].sum()

        # Get the num of pixels from the missed objects
        _, fp_objects, fn_objects = pairwise_object_stats(matches, sum_reduce=False)
        unpaired_true_labels = np.unique(true[true != 0])[fn_objects]
        unpaired_pred_labels = np.unique(pred[pred != 0])[fp_objects]

        for true_id in unpaired_true_labels:
            union += (true == true_id).sum()
        for pred_id in unpaired_pred_labels:
            union += (pred == pred_id).sum()

        # compute aji
        aji = inter / (union + eps)

    return aji


def _absent_inds(true: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Get the class indices that are not present in either `true` or `pred`."""
    t = np.unique(true)
    p = np.unique(pred)
    not_pres = np.setdiff1d(np.arange(num_classes), np.union1d(t, p))

    return not_pres


def iou_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Compute intersection over union (IoU) for each class in a pair of masks.

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses TP / (TP + FP + FN) from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()

    iou = tp / (tp + fp + fn + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        iou[not_pres] = -1.0

    return iou


def accuracy_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Compute pixel accuracy for each class against all other classes.

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses (TP + TN) / pixel_count from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()
    tn = np.prod(true.shape) - (tp + fn + fp)

    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        accuracy[not_pres] = -1.0

    return accuracy


def f1score_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Compute F1 for each class in a pair of class masks.

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses TP / (TP + 0.5 * FP + 0.5 * FN) from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()

    f1 = tp / (0.5 * fp + 0.5 * fn + tp + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        f1[not_pres] = -1.0

    return f1


def dice_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Compute Dice overlap for each class in a pair of class masks.

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses 2 * TP / (2 * TP + FP + FN) from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()

    dice = 2 * tp / (2 * tp + fp + fn + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        dice[not_pres] = -1.0

    return dice


def sensitivity_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Compute sensitivity (recall) for each class in a pair of class masks.

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses TP / (TP + FN) from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()

    sensitivity = tp / (tp + fn + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        sensitivity[not_pres] = -1.0

    return sensitivity


def specificity_multiclass(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
    clamp_absent: bool = True,
) -> np.ndarray:
    """Return per-class precision under the legacy specificity name.

    This function currently calculates precision, TP / (TP + FP). Do not
    use it to measure specificity, which is TN / (TN + FP).

    Known limitation: if a class has no correctly overlapping pixels, the
    underlying pairwise counts are all zero, even when that class appears in
    either mask. Missed and extra pixels can therefore be left out. In particular,
    accuracy can incorrectly return 1. Check this limit before using the scores.

    Args:
        true: Integer HW class mask with IDs from 0 to num_classes-1.
        pred: Same-shaped predicted class IDs, not probabilities or instance IDs.
        num_classes: Total class count, including class 0 if it is background.
        eps: Small offset added to the denominator.
        clamp_absent: Mark a class absent from both masks as -1. False retains
            the computed value. Absent-class markers must be excluded from means.

    Returns:
        Float64 array (num_classes,), ordered by class ID, including class 0.
        Uses TP / (TP + FP) from the current pairwise statistics. No batch or
        dataset mean is calculated. H is height and W width.
    """
    tp, fp, fn = pairwise_pixel_stats(true, pred, num_classes=num_classes)
    tp = tp.diagonal()
    fp = fp.diagonal()
    fn = fn.diagonal()

    specificity = tp / (tp + fp + eps)

    if clamp_absent:
        not_pres = _absent_inds(true, pred, num_classes)
        specificity[not_pres] = -1.0

    return specificity
