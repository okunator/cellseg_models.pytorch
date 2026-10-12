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
    expected_fp = 2
    expected_fn = 1

    iou = pairwise_pixel_stats(
        remap_label(true), remap_label(pred), metric_func=iou_score
    )
    iou = iou[0]
    matches = iou > 0.5

    tp_objects, fp_objects, fn_objects = pairwise_object_stats(matches, sum_reduce=sum_reduce)

    if not sum_reduce:
        assert tp_objects.shape == fn_objects.shape == (4,)
        assert fp_objects.shape == (5,)
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


@pytest.mark.parametrize('thresh', [0.0, 0.25, 0.49])
@pytest.mark.parametrize('transpose', [False, True])
def test_instance_metrics_match_split_and_merge_once(thresh, transpose):
    true = np.array([[0, 1, 1, 1, 1]])
    pred = np.array([[0, 1, 1, 2, 2]])
    if transpose:
        true, pred = pred, true
    quality = panoptic_quality(true, pred, thresh=thresh)
    assert quality['dq'] == pytest.approx(2 / 3)
    assert quality['sq'] == pytest.approx(0.5)
    assert quality['pq'] == pytest.approx(1 / 3)
    expected_precision = 1.0 if transpose else 0.5
    assert average_precision(true, pred, thresh=thresh) == pytest.approx(expected_precision)


@pytest.mark.parametrize('with_background', [False, True])
def test_instance_metrics_accept_sparse_ids(with_background):
    true = np.array([[7, 7, 21, 21]])
    pred = np.array([[42, 42, 3, 3]])
    if with_background:
        true = np.pad(true, ((0, 0), (1, 0)))
        pred = np.pad(pred, ((0, 0), (1, 0)))
    original_true, original_pred = true.copy(), pred.copy()
    for metric in (panoptic_quality, average_precision, aggregated_jaccard_index, dice2):
        result = metric(true, pred)
        if isinstance(result, dict):
            assert all(value == pytest.approx(1.0) for value in result.values())
        else:
            assert result == pytest.approx(1.0)
    np.testing.assert_array_equal(true, original_true)
    np.testing.assert_array_equal(pred, original_pred)


@pytest.mark.parametrize('thresh', [0.5, 1.0])
def test_instance_matching_threshold_is_strict(thresh):
    true = np.array([[0, 1, 1]])
    pred = np.array([[0, 1, 2]])
    assert panoptic_quality(true, pred, thresh=thresh) == {'pq': 0.0, 'sq': 0.0, 'dq': 0.0}
    assert average_precision(true, pred, thresh=thresh) == 0.0


@pytest.mark.parametrize('thresh', [-0.1, 1.1, np.nan, np.inf])
@pytest.mark.parametrize('metric', [panoptic_quality, average_precision])
def test_instance_matching_rejects_invalid_threshold(metric, thresh):
    empty = np.zeros((1, 1), dtype=int)
    with pytest.raises(ValueError, match='thresh'):
        metric(empty, empty, thresh=thresh)


@pytest.mark.parametrize(
    "iou, expected",
    [
        ([[0.65, 0.2], [0.2, 0.0]], [[False, True], [True, False]]),
        ([[0.4, 0.3], [0.3, 0.1]], [[False, True], [True, False]]),
        ([[0.2, 0.3, 0.0]], [[False, True, False]]),
    ],
)
def test_matching_maximizes_pair_count_then_iou(iou, expected):
    from cellseg_models_pytorch.metrics.functional import _match_instances

    np.testing.assert_array_equal(_match_instances(np.array(iou), 0.1), expected)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
@pytest.mark.parametrize("sum_reduce", [False, True])
def test_object_stats_count_empty_sides(shape, sum_reduce):
    result = pairwise_object_stats(np.zeros(shape, dtype=bool), sum_reduce=sum_reduce)
    if not sum_reduce:
        result = tuple(int(values.sum()) for values in result)
    assert result == (0, shape[1], shape[0])


def test_matching_excludes_iou_equal_to_threshold():
    from cellseg_models_pytorch.metrics.functional import _match_instances

    np.testing.assert_array_equal(_match_instances(np.array([[0.5]]), 0.5), [[False]])


@pytest.mark.parametrize("transpose", [False, True])
def test_public_metrics_prioritize_pair_count_over_total_iou(transpose):
    true = np.repeat([7, 21], [125, 25])[None, :]
    pred = np.repeat([42, 3, 42], [100, 25, 25])[None, :]
    if transpose:
        true, pred = pred, true

    # One pair has IoU 2/3; the two alternative pairs each have IoU 1/5.
    low = panoptic_quality(true, pred, thresh=0.1)
    high = panoptic_quality(true, pred, thresh=0.5)
    assert low == pytest.approx({"dq": 1.0, "sq": 1 / 5, "pq": 1 / 5})
    assert high == pytest.approx({"dq": 1 / 2, "sq": 2 / 3, "pq": 1 / 3})
    assert average_precision(true, pred, thresh=0.1) == pytest.approx(1.0)
    assert average_precision(true, pred, thresh=0.5) == pytest.approx(1 / 2)


@pytest.mark.parametrize("transpose", [False, True])
def test_sparse_ids_with_missed_and_extra_instances(transpose):
    true = np.array([[0, 7, 7, 21, 21, 50, 50, 0]])
    pred = np.array([[0, 42, 42, 3, 3, 0, 0, 99]])
    if transpose:
        true, pred = pred, true

    # Four matched pixels, two missed pixels and one extra pixel give AJI 4/7.
    assert aggregated_jaccard_index(true, pred) == pytest.approx(4 / 7)
    assert panoptic_quality(true, pred) == pytest.approx(
        {"dq": 2 / 3, "sq": 1.0, "pq": 2 / 3}
    )
    assert average_precision(true, pred) == pytest.approx(2 / 3)
    iou = pairwise_pixel_stats(true, pred, metric_func=iou_score)[0]
    assert pairwise_object_stats(iou > 0.5) == (2, 1, 1)
