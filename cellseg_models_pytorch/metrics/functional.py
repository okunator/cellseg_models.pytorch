"""
Parts of these functions borrow from the HoVer-Net repo.

https://github.com/vqdang/hover_net/blob/master/metrics/stats_utils.py

MIT License

Copyright (c) 2020 vqdang

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


from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .binary_metrics import get_stats, iou_score

__all__ = [
    "pairwise_pixel_stats",
    "pairwise_object_stats",
    "panoptic_quality",
    "average_precision",
    "aggregated_jaccard_index",
    "dice2",
]


def pairwise_pixel_stats(
    true: np.ndarray, pred: np.ndarray, metric_func: Optional[Callable] = None
) -> Union[List[np.ndarray], None]:
    """Compute the number of TP, FP, FN pixels for each object in a labelled mask.

    Optionally a binary metric can be computed instead of the satistics.
    Atleast 2x faster than computing with `np.histogram2d`.

    Parameters
    ----------
        true : np.ndarray
            Ground truth (labelled mask). Shape (H, W).
        pred : np.ndarray
            Predicted (labelled mask). Shape (H, W).
        metric_func : Callable, optional
            A binary metric function. e.g. `iou_score` or `dice`.

    Returns
    -------
        List[np.ndarray, ...] or None:
            A List of 2D arrays (i, j) where i corresponds to a ground
            truth label and j corresponds to a predicted label. Each value
            of the matrix is the computed statistic or metric at pos (i, j).

            Shape: (n_labels_gt, n_labels_pred). Dtype. float64.
    """
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

            # ignore bg and empty preds
            if pred_label == 0:
                continue

            pred_obj = pred_objects[pred_label]
            tp, fp, fn = get_stats(true_obj, pred_obj)

            # compute a metric or add stats
            if metric_func is not None:
                ret[0][true_label - 1, pred_label - 1] = metric_func(tp, fp, fn)
            else:
                ret[0][true_label - 1, pred_label - 1] = tp.sum()
                ret[1][true_label - 1, pred_label - 1] = fp.sum()
                ret[2][true_label - 1, pred_label - 1] = fn.sum()

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
        thresh (float, default=0.5):
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        Dict[str, float]:
            Dictionary containing the detection quality (dq), segmentation
            quality (sq) and panoptic quality (pq) values.
    """
    iou = pairwise_pixel_stats(true, pred, iou_score)
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
        thresh (float, default=0.5):
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        float:
            The compuited precision.
    """
    iou = pairwise_pixel_stats(pred, true, iou_score)
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
        thresh (float, default=0.5):
            Threshold for the iou to include the prediction as TP
        eps : float, default=1e-8:
            Epsilon to avoid zero div errors.

    Returns
    -------
        float:
            The computed precision.
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
