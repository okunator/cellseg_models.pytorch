from typing import Callable, Dict, List, Tuple, Union

import numpy as np

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
    num_classes: int = None,
    metric_func: Callable = None,
) -> Union[List[np.ndarray], None]:
    """Compute the # of TP, FP, FN pixels for each object in a labelled/semantic mask.

    Optionally a binary metric can be computed instead of the satistics.
    Atleast 2x faster than computing with `np.histogram2d`.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (semantic or labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (semantic or labelled mask). Shape (H, W).
        num_classes : int, optional
            Number of classes in the dataset. If None, stats are computed for instances.
            If not None stats are computed for classes i.e. semantic segmentation masks.
        metric_func : Callable, optional
            A binary metric function. e.g. `iou_score` or `dice`.

    Returns
    -------
        List[np.ndarray, ...] or None:
            A List of 2D arrays (i, j) where i corresponds to a ground
            truth label and j corresponds to a predicted label. Each value
            of the matrix is the computed statistic or metric at pos (i, j).
            By default. returns the tp, fp, and fn matrices.

            If stats computed for instances:
                Shape: (n_labels_gt, n_labels_pred). Dtype. float64.
            If stats computed for classes:
                Shape: (num_classes, num_classes). Dtype. float64.
    """
    if num_classes is not None:
        true_labels = list(range(num_classes))
        pred_labels = list(range(num_classes))
    else:
        true_labels = list(np.unique(true))[1:]
        pred_labels = list(np.unique(pred))[1:]

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

            if num_classes is None:
                ix = true_label - 1
                jx = pred_label - 1
            else:
                ix = true_label
                jx = pred_label

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
) -> Union[Tuple[int, int, int], Tuple[List[bool]]]:
    """Compute the TP, FP, FN objects from a boolean contigency table.

    Parameters
    ----------
        matches : np.ndarray
            A pairwise boolean matrix where True values at pos (i, j)
            indicate correctly detected objects for the corresponding
            labels i and j. Shape: (n_labels_gt, n_labels_pred).
        sum_reduce : bool, default=True
            Reduce the boolean indice arrays by summing to get the correct
            number of TP, FP, and FN objects.

    Returns
    -------
        Tuple[int, int, int]:
            The number of TP objects, FP objects, and FN objects in
            a labelled mask.

    """
    true_hits = matches.sum(axis=0)
    pred_hits = matches.sum(axis=1)

    tp_objects = pred_hits >= 1  # indices of correctly predicted objects
    fp_objects = pred_hits == 0  # indices of missed objects in prediciton
    fn_objects = true_hits == 0  # indices of extra objects in prediction

    if sum_reduce:
        tp_objects = tp_objects.sum()
        fp_objects = fp_objects.sum()
        fn_objects = fn_objects.sum()

    return tp_objects, fp_objects, fn_objects


def panoptic_quality(
    true: np.ndarray, pred: np.ndarray, thresh: float = 0.5, eps: float = 1e-8
) -> Dict[str, float]:
    """Compute the panoptic quality of a lebelled mask.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (labelled mask). Shape (H, W).
        thresh : float, default=0.5
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        Dict[str, float]:
            Dictionary containing the detection quality (dq), segmentation
            quality (sq) and panoptic quality (pq) values.
    """
    iou = pairwise_pixel_stats(true, pred, metric_func=iou_score)
    res = {"pq": 0.0, "sq": 0.0, "dq": 0.0}

    if iou:
        iou = iou[0]
        matches = iou > thresh
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
    """Compute the average precision of a labelled mask.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (labelled mask). Shape (H, W).
        thresh : float, default=0.5
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        float:
            The computed precision.
    """
    iou = pairwise_pixel_stats(pred, true, metric_func=iou_score)
    ap = 0.0

    if iou:
        iou = iou[0]
        matches = iou > thresh
        tp_objects, fp_objects, _ = pairwise_object_stats(matches)

        ap = tp_objects / (tp_objects + fp_objects + eps)

    return ap


def dice2(true: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the DICE2 metric for a labelled mask.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (labelled mask). Shape (H, W).
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        float:
            The computed dice2 metric.

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
    """Compute the aggregated jaccard index (AJI) for a labelled mask.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (labelled mask). Shape (H, W).
        thresh : float, default=0.5
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        float:
            The computed aji.
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
        unpaired_true_labels = np.nonzero(fp_objects)[0] + 1
        unpaired_pred_labels = np.nonzero(fn_objects)[0] + 1

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
    """Compute multi-class intersection over union for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent : bool, default=True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class IoU-metrics. Shape: (num_classes,).
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
    """Compute multi-class accuracy for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent: bool = True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class accuracy-metrics. Shape: (num_classes,).
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
    """Compute multi-class f1-score for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent: bool = True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class f1score-metrics. Shape: (num_classes,).
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
    """Compute multi-class dice for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent: bool = True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class dice-metrics. Shape: (num_classes,).
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
    """Compute multi-class sensitivity for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent: bool = True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class sensitivity-metrics. Shape: (num_classes,).
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
    """Compute multi-class specificity for semantic segmentation masks.

    Parameters
    ----------
        true : np.ndarray
            Ground truth semantic mask. Shape (H, W).
        pred : np.ndarray
            Predicted semantic mask. Shape (H, W).
        num_classes : int
            Number of classes in the training dataset.
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.
        clamp_absent: bool = True
            If a class is not present in either true or pred, the value of that ix
            in the result array will be clamped to -1.0.

    Returns
    -------
        np.ndarray:
            Per class specificity-metrics. Shape: (num_classes,).
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
