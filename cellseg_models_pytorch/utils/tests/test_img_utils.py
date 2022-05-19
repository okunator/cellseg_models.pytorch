import numpy as np
import pytest

from cellseg_models_pytorch.utils import (
    float2ubyte,
    minmax_normalize,
    normalize,
    percentile_normalize,
    percentile_normalize99,
)


@pytest.mark.parametrize("lower", [0.0, 0.1])
@pytest.mark.parametrize("upper", [99.99, 100.0])
def test_percentile_normalize(img_sample, lower, upper) -> None:
    im = img_sample
    nim = percentile_normalize(im, lower, upper)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape
    assert np.logical_and(nim >= 0.0, nim <= 1.0).all()


@pytest.mark.parametrize("amin", [None, 0.0])
@pytest.mark.parametrize("amax", [1.0, None])
def test_percentile_normalize99(img_sample, amin, amax) -> None:
    im = img_sample
    nim = percentile_normalize99(im, amin, amax)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape


@pytest.mark.parametrize("amin", [None, 0.0])
@pytest.mark.parametrize("amax", [1.0, None])
@pytest.mark.parametrize("standardize", [True, False])
def test_normalize(img_sample, standardize, amin, amax) -> None:
    im = img_sample
    nim = normalize(im, standardize, amin, amax)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape


@pytest.mark.parametrize("amin", [None, 0.0])
@pytest.mark.parametrize("amax", [1.0, None])
def test_minmax_normalize(img_sample, amin, amax) -> None:
    im = img_sample
    nim = minmax_normalize(im, amin, amax)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape
    assert np.logical_and(nim >= 0.0, nim <= 1.0).all()


@pytest.mark.parametrize("norm", [True, False])
def test_float2ubyte(img_sample, norm) -> None:
    im = img_sample
    nim = normalize(im)
    ubyte = float2ubyte(nim, norm)

    assert ubyte.dtype == "uint8"
    assert ubyte.shape == im.shape
    assert np.logical_and(ubyte >= 0, ubyte <= 255).all()
