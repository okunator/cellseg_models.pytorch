import numpy as np
import pytest

from cellseg_models_pytorch.metrics import (
    accuracy_multiclass,
    dice_multiclass,
    f1score_multiclass,
    iou_multiclass,
    sensitivity_multiclass,
    specificity_multiclass,
)


@pytest.mark.parametrize("zero_true", [False, True])
@pytest.mark.parametrize("zero_pred", [False, True])
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
def test_semantic_metrics(true_sem, pred_sem, metric, zero_pred, zero_true):
    if zero_pred:
        pred_sem = np.zeros_like(true_sem)
    if zero_true:
        true_sem = np.zeros_like(pred_sem)

    metrics = metric(true_sem, pred_sem, num_classes=5)

    assert metrics.shape == (5,)

    if not zero_pred and not zero_true:
        assert metrics[2] != 1.0
