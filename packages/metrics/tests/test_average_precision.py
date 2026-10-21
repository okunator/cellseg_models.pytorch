"""Independent precision–recall examples for confidence-ranked instance AP."""

import numpy as np
import pytest

from cellseg_metrics import mean_average_precision, precision_recall_ap

METRICS = [precision_recall_ap, mean_average_precision]


@pytest.mark.parametrize("metric", METRICS)
def test_perfect_masks_with_sparse_ids_and_no_background(metric):
    true = np.array([[7, 7, 21, 21]], dtype=np.uint64)
    pred = np.array([[10**12, 10**12, 3, 3]], dtype=np.uint64)
    originals = true.copy(), pred.copy()

    assert metric(true[:, ::-1], pred[:, ::-1], {3: 0.1, 10**12: 0.9}) == 1.0
    np.testing.assert_array_equal(true, originals[0])
    np.testing.assert_array_equal(pred, originals[1])


@pytest.mark.parametrize(
    "metric,expected",
    [(precision_recall_ap, 4 / 9), (mean_average_precision, 134 / 303)],
)
def test_precision_envelope_and_missed_reference(metric, expected):
    # Ranked outcomes: FP, TP, TP. Two of three reference nuclei are detected.
    true = np.array([[0, 7, 7, 0, 21, 21, 0, 41, 41]])
    pred = np.array([[9, 3, 3, 0, 13, 13, 0, 0, 0]])

    score = metric(true, pred, {9: 0.9, 3: 0.8, 13: 0.7})

    # All-points AP = (2/3 recall) * (2/3 precision). 101-point AP uses 67 samples.
    assert score == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS)
def test_confidence_order_matters_and_late_false_positives_do_not_reduce_ap(metric):
    true = np.array([[0, 7, 7, 21, 21]])
    pred = np.array([[9, 3, 3, 13, 13]])

    assert metric(true, pred, {9: 0.1, 3: 0.9, 13: 0.8}) == 1.0
    assert metric(true, pred, {9: 0.9, 3: 0.8, 13: 0.7}) == pytest.approx(2 / 3)


@pytest.mark.parametrize("metric", METRICS)
def test_global_ranking_includes_false_detections_on_empty_images(metric):
    true = [np.array([[7, 7]]), np.zeros((2, 3), dtype=int)]
    pred = [np.array([[3, 3]]), np.array([[3, 3, 0], [0, 0, 0]])]

    # The empty-reference image supplies the highest-confidence false detection.
    assert metric(true, pred, [{3: 0.5}, {3: 0.9}]) == 0.5


@pytest.mark.parametrize("metric", METRICS)
def test_reference_ids_are_local_to_each_image(metric):
    true = [np.array([[7, 7]]), np.array([[7, 7]])]
    pred = [np.array([[3, 3]]), np.array([[3, 3]])]

    assert metric(true, pred, [{3: -5.0}, {3: 4.0}]) == 1.0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize(
    "lower,higher",
    [
        (2**53, 2**53 + 1),
        (np.int64(-(2**63)), np.int64(-(2**63) + 1)),
        (np.uint64(2**64 - 2), np.uint64(2**64 - 1)),
        (np.float64(2**53), 2**53 + 1),
        (np.float32(2**24), 2**24 + 1),
    ],
)
def test_confidence_order_is_preserved_without_lossy_numeric_promotion(
    metric, dataset, lower, higher
):
    if dataset:
        # A tie introduced during global ranking would put the empty image's FP first.
        true = [np.array([[0]]), np.array([[7]])]
        pred = [np.array([[3]]), np.array([[9]])]
        scores = [{3: lower}, {9: higher}]
    else:
        # A tie introduced before matching would put the smaller prediction ID first.
        true = np.array([[0, 7, 7]])
        pred = np.array([[3, 9, 9]])
        scores = {3: lower, 9: higher}

    assert metric(true, pred, scores) == 1.0


@pytest.mark.parametrize("metric", METRICS)
def test_confidence_ties_use_image_order_then_sorted_prediction_id(metric):
    true = np.array([[0, 7, 7]])
    pred = np.array([[9, 3, 3]])

    assert metric(true, pred, {9: 0.5, 3: 0.5}) == 1.0
    # The first image's false detection wins an equal-score tie across images.
    assert (
        metric(
            [np.zeros_like(true), true],
            [np.array([[9, 0, 0]]), pred],
            [{9: 0.5}, {3: 0.5, 9: 0.5}],
        )
        == 0.5
    )


@pytest.mark.parametrize(
    "metric,kwargs,expected",
    [
        (precision_recall_ap, {"iou_threshold": 0.2}, 0.5),
        (mean_average_precision, {"iou_thresholds": [0.2]}, 51 / 101),
    ],
)
def test_greedy_matching_uses_confidence_not_maximum_total_pair_count(
    metric, kwargs, expected
):
    true = np.array([[1, 1, 1, 1, 2, 2, 2, 2]])
    pred = np.array([[3, 3, 3, 4, 3, 3, 0, 0]])

    # Prediction 3 takes reference 1 at IoU 0.5; prediction 4 cannot reuse it.
    assert metric(true, pred, {3: 0.9, 4: 0.8}, **kwargs) == pytest.approx(expected)
    # In reverse order, prediction 3 must select its next-best unmatched reference.
    assert metric(true, pred, {3: 0.8, 4: 0.9}, **kwargs) == 1.0


@pytest.mark.parametrize(
    "metric,kwargs,one_match",
    [
        (precision_recall_ap, {"iou_threshold": 0.3}, 0.5),
        (mean_average_precision, {"iou_thresholds": [0.3]}, 51 / 101),
    ],
)
@pytest.mark.parametrize("reference_ids", [(7, 21), (21, 7)])
def test_equal_iou_matches_favor_larger_reference_id(
    metric, kwargs, one_match, reference_ids
):
    left, right = reference_ids
    true = np.array([[left] * 4 + [right] * 4])
    pred = np.array([[3, 3, 0, 0, 3, 3, 4, 4]])
    # Prediction 3 overlaps both references by 1/3; prediction 4 overlaps only the right.
    expected = one_match if right > left else 1.0

    assert metric(true, pred, {3: 0.9, 4: 0.8}, **kwargs) == pytest.approx(expected)


def test_iou_boundaries_are_inclusive_without_denominator_epsilon():
    true = np.array([[7, 7, 7, 7]])
    pred = np.array([[3, 3, 9, 9]])

    assert precision_recall_ap(true, pred, {3: 0.9, 9: 0.8}) == 1.0
    assert mean_average_precision(true, pred, {3: 0.9, 9: 0.8}) == 0.1
    assert precision_recall_ap(true, true, {7: 0.9}, iou_threshold=1) == 1.0


def test_mean_ap_recomputes_matches_at_every_iou_and_uses_101_recall_points():
    true = np.array([[1] * 10 + [2] * 10])
    pred = np.array([[3] * 10 + [9] * 7 + [0] * 3])

    # Both match at 0.50–0.70. Above that, recall 0.5 fills 51 of 101 recall bins.
    assert mean_average_precision(true, pred, {3: 0.9, 9: 0.8}) == pytest.approx(
        76 / 101
    )
    assert mean_average_precision(
        true, pred, {3: 0.9, 9: 0.8}, iou_thresholds=[1.0, 0.5]
    ) == pytest.approx(76 / 101)
    assert precision_recall_ap(true, pred, {3: 0.9, 9: 0.8}, 0.75) == 0.5


@pytest.mark.parametrize("metric", METRICS)
def test_all_nuclei_are_used_without_coco_detection_limit(metric):
    mask = np.arange(1, 152).reshape(1, -1)

    assert metric(mask, mask, dict.fromkeys(range(1, 152), 0.5)) == 1.0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "case", ["no_predictions", "no_overlap", "no_reference", "empty"]
)
def test_empty_and_unmatched_inputs(metric, case):
    true, pred = np.array([[7, 7, 0, 0]]), np.array([[0, 0, 3, 3]])
    if case in ("no_predictions", "empty"):
        pred[:] = 0
    if case in ("no_reference", "empty"):
        true[:] = 0
    scores = {3: 0.8} if np.any(pred) else {}

    result = metric(true, pred, scores)

    if case in ("no_reference", "empty"):
        assert np.isnan(result)
    else:
        assert result == 0.0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "scores",
    [{}, {0: 0.9, 3: 0.8}, {3: np.nan}, {3: np.inf}, {3: 1j}, {3: "high"}, {3: [0.5]}],
)
def test_invalid_confidence_mappings(metric, scores):
    with pytest.raises(ValueError, match="Score|Confidence"):
        metric(np.array([[7]]), np.array([[3]]), scores)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "true,pred",
    [
        (np.array([1]), np.array([1])),
        (np.array([[1]]), np.array([[1, 1]])),
        (np.array([[1.0]]), np.array([[1]])),
        (np.array([[1]]), np.array([[True]])),
        (np.array([[-1]]), np.array([[1]])),
        (np.array([[1]]), np.array([[-1]])),
    ],
)
def test_invalid_instance_masks(metric, true, pred):
    with pytest.raises(ValueError, match="Instance"):
        metric(true, pred, {1: 0.9})


@pytest.mark.parametrize("metric", METRICS)
def test_image_count_and_single_dataset_mismatches(metric):
    mask = np.array([[1]])
    with pytest.raises(ValueError, match="image counts"):
        metric([mask, mask], [mask], [{1: 0.9}])
    with pytest.raises(ValueError, match="sequences"):
        metric(mask, [mask], [{1: 0.9}])
    with pytest.raises(ValueError, match="sequences"):
        metric([mask], mask, {1: 0.9})


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_iou_thresholds(threshold):
    mask = np.array([[1]])
    with pytest.raises(ValueError, match="IoU thresholds"):
        precision_recall_ap(mask, mask, {1: 0.9}, threshold)
    with pytest.raises(ValueError, match="IoU thresholds"):
        mean_average_precision(mask, mask, {1: 0.9}, [threshold])


@pytest.mark.parametrize("thresholds", [[], [0.5, 0.5], [[0.5]]])
def test_invalid_iou_threshold_sequences(thresholds):
    mask = np.array([[1]])
    with pytest.raises(ValueError, match="IoU thresholds"):
        mean_average_precision(mask, mask, {1: 0.9}, thresholds)
