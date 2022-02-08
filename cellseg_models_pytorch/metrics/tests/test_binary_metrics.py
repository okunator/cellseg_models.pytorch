import numpy as np
import pytest

from cellseg_models_pytorch.metrics import (
    accuracy,
    dice_coef,
    f1score,
    get_stats,
    iou_score,
)


@pytest.fixture(scope="package")
def binary_true() -> np.ndarray:
    true = true = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ],
        dtype=int,
    )

    return true


@pytest.fixture(scope="package")
def binary_pred() -> np.ndarray:
    pred = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    return pred


def test_get_stats(binary_true, binary_pred):
    tp, fp, fn = get_stats(binary_true, binary_pred)

    assert tp.sum() == 32
    assert fp.sum() == 4
    assert fn.sum() == 2


def test_iou(binary_true, binary_pred):
    tp, fp, fn = get_stats(binary_true, binary_pred)
    iou = iou_score(tp, fp, fn)

    expected = 0.84
    np.testing.assert_allclose(iou, expected, rtol=1e-2, err_msg=f"{iou} != {expected}")


def test_accuracy(binary_true, binary_pred):
    tp, fp, fn = get_stats(binary_true, binary_pred)
    acc = accuracy(tp, fp, fn)

    expected = 0.90
    np.testing.assert_allclose(acc, expected, rtol=1e-2, err_msg=f"{acc} != {expected}")


def test_dice(binary_true, binary_pred):
    tp, fp, fn = get_stats(binary_true, binary_pred)
    dice = dice_coef(tp, fp, fn)

    expected = 0.91
    np.testing.assert_allclose(
        dice, expected, rtol=1e-2, err_msg=f"{dice} != {expected}"
    )


def test_f1score(binary_true, binary_pred):
    tp, fp, fn = get_stats(binary_true, binary_pred)
    f1 = f1score(tp, fp, fn)

    expected = 0.91
    np.testing.assert_allclose(f1, expected, rtol=1e-2, err_msg=f"{f1} != {expected}")
