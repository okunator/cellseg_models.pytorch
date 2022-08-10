import numpy as np
import pytest

from cellseg_models_pytorch.metrics import (
    aggregated_jaccard_index,
    average_precision,
    dice2,
    iou_score,
    pairwise_object_stats,
    pairwise_pixel_stats,
    panoptic_quality,
)
from cellseg_models_pytorch.utils import remap_label


@pytest.fixture(scope="package")
def true() -> np.ndarray:
    true = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 4, 4, 4, 4],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 0],
            [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    return true


@pytest.fixture(scope="package")
def pred() -> np.ndarray:
    pred = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
            [0, 4, 4, 4, 4, 0, 0, 3, 3, 3, 3, 0, 0, 0],
            [4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 0, 0],
            [4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 0, 0],
            [0, 4, 4, 4, 4, 0, 0, 3, 3, 3, 3, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 5, 6, 6, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 5, 5, 6, 6, 6, 0, 0, 0, 0, 0],
            [0, 0, 5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 5, 6, 6, 6, 6, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    return pred


@pytest.mark.parametrize("metric_func", [None, iou_score])
def test_get_pixel_stats(true, pred, metric_func):
    expected_shape = (len(np.unique(true)[1:]), len(np.unique(pred)[1:]))
    stats = pairwise_pixel_stats(
        remap_label(true), remap_label(pred), metric_func=metric_func
    )[0]

    assert stats.shape == expected_shape


@pytest.mark.parametrize("sum_reduce", [True, False])
def test_pairwise_object_stats(true, pred, sum_reduce):
    expected_tp = 3
    expected_fp = 1
    expected_fn = 2

    iou = pairwise_pixel_stats(
        remap_label(true), remap_label(pred), metric_func=iou_score
    )
    iou = iou[0]
    matches = iou > 0.5

    tp_objects, fp_objects, fn_objects = pairwise_object_stats(matches)

    if sum_reduce:
        tp_objects = tp_objects.sum()
        fp_objects = fp_objects.sum()
        fn_objects = fn_objects.sum()

    assert tp_objects == expected_tp
    assert fp_objects == expected_fp
    assert fn_objects == expected_fn


@pytest.mark.parametrize("true_zeros", [None, np.zeros((14, 14))])
@pytest.mark.parametrize("pred_zeros", [None, np.zeros((14, 14))])
def test_panoptic_quality(true, pred, true_zeros, pred_zeros):

    if true_zeros is not None:
        true = true_zeros

    if pred_zeros is not None:
        pred = pred_zeros

    metrics = panoptic_quality(remap_label(true), remap_label(pred))

    expected = 0.0
    if pred_zeros is None and true_zeros is None:
        expected = 0.51

    np.testing.assert_allclose(
        metrics["pq"], expected, rtol=1e-2, err_msg=f"{metrics['pq']} != {expected}"
    )


@pytest.mark.parametrize("true_zeros", [None, np.zeros((14, 14))])
@pytest.mark.parametrize("pred_zeros", [None, np.zeros((14, 14))])
def test_average_precision(true, pred, true_zeros, pred_zeros):

    if true_zeros is not None:
        true = true_zeros

    if pred_zeros is not None:
        pred = pred_zeros

    ap = average_precision(remap_label(true), remap_label(pred))

    expected = 0.0
    if pred_zeros is None and true_zeros is None:
        expected = 0.6

    np.testing.assert_allclose(ap, expected, rtol=1e-2, err_msg=f"{ap} != {expected}")


@pytest.mark.parametrize("true_zeros", [None, np.zeros((14, 14))])
@pytest.mark.parametrize("pred_zeros", [None, np.zeros((14, 14))])
def test_dice2(true, pred, true_zeros, pred_zeros):

    if true_zeros is not None:
        true = true_zeros

    if pred_zeros is not None:
        pred = pred_zeros

    dice = dice2(remap_label(true), remap_label(pred))

    expected = 0.0
    if pred_zeros is None and true_zeros is None:
        expected = 0.64

    np.testing.assert_allclose(
        dice, expected, rtol=1e-2, err_msg=f"{dice} != {expected}"
    )


@pytest.mark.parametrize("true_zeros", [None, np.zeros((14, 14))])
@pytest.mark.parametrize("pred_zeros", [None, np.zeros((14, 14))])
def test_aji(true, pred, true_zeros, pred_zeros):

    if true_zeros is not None:
        true = true_zeros

    if pred_zeros is not None:
        pred = pred_zeros

    aji = aggregated_jaccard_index(remap_label(true), remap_label(pred))

    expected = 0.0
    if pred_zeros is None and true_zeros is None:
        expected = 0.53

    np.testing.assert_allclose(aji, expected, rtol=1e-2, err_msg=f"{aji} != {expected}")
