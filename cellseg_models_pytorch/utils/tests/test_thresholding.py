import pytest
import numpy as np
from typing import Callable

from cellseg_models_pytorch.utils import (
    naive_thresh,
    naive_thresh_prob,
    sauvola_thresh,
    morph_chan_vese_thresh,
    smoothed_thresh,
    niblack_thresh,
    argmax,
)


@pytest.fixture
def prob_map() -> np.ndarray:
    return np.random.rand(256, 256)


@pytest.mark.parametrize(
    "method",
    [
        naive_thresh,
        naive_thresh_prob,
        sauvola_thresh,
        morph_chan_vese_thresh,
        smoothed_thresh,
        niblack_thresh,
    ],
)
def test_thresh(prob_map, method):
    """
    Quick tests for the different thresholding methods
    """
    binary = method(prob_map)

    if len(np.unique(binary)) == 2:
        assert np.amax(binary) == 1
        assert np.amin(binary) == 0

    assert binary.dtype == "uint8"


def test_argmax(prob_map):
    observed = argmax(prob_map)
    assert observed.shape == (256, 256)
    assert observed.dtype == "uint32"
