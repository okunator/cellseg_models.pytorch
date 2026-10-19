"""Compare class-mask metrics with independent confusion-matrix counts."""

import numpy as np
import pytest

from cellseg_models_pytorch.metrics import (
    accuracy,
    accuracy_multiclass,
    dice_multiclass,
    f1score_multiclass,
    iou_multiclass,
    iou_score,
    pairwise_pixel_stats,
    sensitivity_multiclass,
    specificity_multiclass,
)


@pytest.fixture(params=["mixed", "swapped", "background"])
def class_masks(request):
    if request.param == "mixed":
        # Classes 3 and 4 occur on only one side; class 5 is absent from both.
        true = np.array([[0, 0, 0, 0, 1], [1, 2, 2, 3, 3]], dtype=np.uint8)
        pred = np.array([[0, 0, 1, 4, 2], [2, 0, 2, 0, 2]], dtype=np.uint8)
        confusion = np.array(
            [
                [2, 1, 0, 0, 1, 0],
                [0, 0, 2, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    elif request.param == "swapped":
        true = np.array([[0, 0, 1, 1]], dtype=np.int32)
        pred = np.array([[1, 1, 0, 0]], dtype=np.int32)
        confusion = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]])
    else:
        true = np.zeros((2, 3), dtype=np.int64)
        pred = np.zeros_like(true)
        confusion = np.array([[6, 0], [0, 0]])

    return true, pred, confusion


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("metric_func", [None, iou_score, accuracy])
def test_class_pair_counts_and_callbacks_match_confusion_matrix(
    class_masks, transpose, metric_func
):
    true, pred, confusion = class_masks
    if transpose:
        true, pred = true.T, pred.T
    originals = true.copy(), pred.copy()
    num_classes = confusion.shape[0]

    # Rows count reference classes; columns count predicted classes.
    tp = confusion.astype(np.float64)
    fp = confusion.sum(axis=0)[None, :] - tp
    fn = confusion.sum(axis=1)[:, None] - tp
    if metric_func is None:
        expected = [tp, fp, fn]
    elif metric_func is iou_score:
        expected = [tp / (tp + fp + fn + 1e-8)]
    else:
        tn = confusion.sum() - tp - fp - fn
        expected = [(tp + tn) / (confusion.sum() + 1e-8)]

    result = pairwise_pixel_stats(true, pred, num_classes, metric_func)
    assert len(result) == len(expected)
    for actual, reference in zip(result, expected):
        assert actual.shape == (num_classes, num_classes)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(true, originals[0])
    np.testing.assert_array_equal(pred, originals[1])


@pytest.mark.parametrize("clamp_absent", [False, True])
@pytest.mark.parametrize("eps", [1e-8, 0.25])
@pytest.mark.parametrize(
    "metric",
    [
        iou_multiclass,
        dice_multiclass,
        f1score_multiclass,
        accuracy_multiclass,
        sensitivity_multiclass,
        specificity_multiclass,
    ],
)
def test_class_scores_match_confusion_matrix(class_masks, clamp_absent, eps, metric):
    true, pred, confusion = class_masks
    tp = confusion.diagonal()
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    tn = confusion.sum() - tp - fp - fn
    reference = {
        iou_multiclass: tp / (tp + fp + fn + eps),
        dice_multiclass: 2 * tp / (2 * tp + fp + fn + eps),
        f1score_multiclass: tp / (tp + 0.5 * (fp + fn) + eps),
        accuracy_multiclass: (tp + tn) / (confusion.sum() + eps),
        sensitivity_multiclass: tp / (tp + fn + eps),
        specificity_multiclass: tn / (tn + fp + eps),
    }[metric]
    if clamp_absent:
        absent = confusion.sum(axis=0) + confusion.sum(axis=1) == 0
        reference[absent] = -1

    actual = metric(true, pred, confusion.shape[0], eps=eps, clamp_absent=clamp_absent)
    assert actual.shape == (confusion.shape[0],)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=0)


def test_specificity_uses_true_negatives_not_precision():
    true = np.array([[1, 1, 0, 0, 0]])
    pred = np.array([[1, 1, 1, 1, 0]])
    # Class 1: TP=2, FP=2, FN=0, TN=1. Precision would be 1/2.
    actual = specificity_multiclass(true, pred, num_classes=2, eps=0)
    np.testing.assert_allclose(actual, [1, 1 / 3], rtol=1e-12, atol=0)


@pytest.mark.parametrize("metric_func", [None, iou_score])
def test_instance_pairs_still_skip_nonoverlapping_objects(metric_func):
    true = np.array([[0, 7, 7, 21, 21, 50, 50, 0]])
    pred = np.array([[0, 42, 42, 3, 3, 0, 0, 99]])
    overlap = np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.float64)
    if metric_func is None:
        expected = [overlap, np.zeros((3, 3)), np.zeros((3, 3))]
    else:
        expected = [overlap / (2 + 1e-8)]

    actual = pairwise_pixel_stats(true, pred, metric_func=metric_func)
    assert len(actual) == len(expected)
    for counts, reference in zip(actual, expected):
        np.testing.assert_allclose(counts, reference, rtol=1e-12, atol=0)


@pytest.mark.parametrize("empty_side", ["true", "pred", "both"])
def test_instance_pairs_keep_empty_result_when_either_side_has_no_objects(empty_side):
    true = np.array([[0, 7]])
    pred = np.array([[0, 42]])
    if empty_side in ("true", "both"):
        true.fill(0)
    if empty_side in ("pred", "both"):
        pred.fill(0)
    assert pairwise_pixel_stats(true, pred) == []
