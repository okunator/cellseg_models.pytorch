import numpy as np
import pytest

from cellseg_models_pytorch.utils import (
    argmax,
    morph_chan_vese_thresh,
    naive_thresh,
    naive_thresh_prob,
    niblack_thresh,
    sauvola_thresh,
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
        niblack_thresh,
    ],
)
def test_thresh(prob_map, method):
    binary = method(prob_map)

    if len(np.unique(binary)) == 2:
        assert np.amax(binary) == 1
        assert np.amin(binary) == 0

    assert binary.dtype == "uint8"


def test_argmax(prob_map):
    observed = argmax(prob_map)
    assert observed.shape == (256, 256)
    assert observed.dtype == "uint32"
