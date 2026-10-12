import numpy as np
import pytest

from cellseg_models_pytorch.transforms.functional.normalization import (
    float2ubyte,
    minmax_normalize,
    normalize,
    percentile_normalize,
    percentile_normalize99,
)


@pytest.mark.parametrize("lower", [0.0, 0.1])
@pytest.mark.parametrize("upper", [99.99, 100.0])
@pytest.mark.parametrize("copy", [True, False])
def test_percentile_normalize(img_sample, lower, upper, copy) -> None:
    im = img_sample.astype(np.float32)
    nim = percentile_normalize(im, lower, upper, copy)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape
    assert np.logical_and(nim >= 0.0, nim <= 1.0).all()


@pytest.mark.parametrize("amin", [None, 0.0])
@pytest.mark.parametrize("amax", [1.0, None])
@pytest.mark.parametrize("copy", [True, False])
def test_percentile_normalize99(img_sample, amin, amax, copy) -> None:
    im = img_sample.astype(np.float32)
    nim = percentile_normalize99(im, amin, amax, copy)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    ("amin", "amax", "expected"),
    [
        (None, None, [[-1 / 98, 97 / 294], [197 / 294, 99 / 98]]),
        (0, 1, [[0, 97 / 294], [197 / 294, 1]]),
        (0.5, None, [[0.5, 0.5], [197 / 294, 99 / 98]]),
        (None, 0.5, [[-1 / 98, 97 / 294], [0.5, 0.5]]),
    ],
)
def test_percentile_normalize99_values_and_clipping(amin, amax, expected, copy):
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    original = image.copy()

    actual = percentile_normalize99(image, amin=amin, amax=amax, copy=copy)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 3)])
def test_percentile_normalize99_zero_span(monkeypatch, copy, shape):
    divide = np.divide

    def divide_with_poisoned_output(numerator, denominator, **kwargs):
        # Simulate nonzero memory in an uninitialized ufunc output.
        if kwargs.get("out") is None:
            kwargs["out"] = np.full_like(numerator, np.nan)
        return divide(numerator, denominator, **kwargs)

    monkeypatch.setattr(np, "divide", divide_with_poisoned_output)
    image = np.full(shape, 17, dtype=np.uint8)
    expected = np.zeros(shape, dtype=np.float32)
    if len(shape) == 3:
        image[..., 1] = np.arange(4).reshape(2, 2)
        expected[..., 1] = [[-1 / 98, 97 / 294], [197 / 294, 99 / 98]]
    original = image.copy()

    with np.errstate(divide="raise", invalid="raise"):
        actual = percentile_normalize99(image, copy=copy)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(image, original)


def test_percentile_normalize99_clips_zero_span_to_lower_bound():
    image = np.zeros((1, 101), dtype=np.float32)
    image[0, -1] = 1

    actual = percentile_normalize99(image, amin=0.25, amax=1)

    np.testing.assert_array_equal(actual, np.full(image.shape, 0.25, np.float32))


def test_percentile_normalize99_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="amin must be less than or equal to amax"):
        percentile_normalize99(np.zeros((2, 2)), amin=1, amax=0)


@pytest.mark.parametrize("shape", [(4,), (1, 2, 2, 3)])
def test_percentile_normalize99_rejects_invalid_axes(shape):
    with pytest.raises(ValueError, match="Input img needs to have shape"):
        percentile_normalize99(np.zeros(shape))


@pytest.mark.parametrize("mean", [np.array([35.0, 150.0, 230.0]), None])
@pytest.mark.parametrize("denom", [np.array([35.0, 150.0, 230.0]), None])
@pytest.mark.parametrize("copy", [True, False])
def test_normalize(img_sample, mean, denom, copy) -> None:
    im = img_sample.astype(np.float32)
    nim = normalize(im, mean, denom, copy)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape


@pytest.mark.parametrize("amin", [None, 0.0])
@pytest.mark.parametrize("amax", [1.0, None])
@pytest.mark.parametrize("copy", [True, False])
def test_minmax_normalize(img_sample, amin, amax, copy) -> None:
    im = img_sample.astype(np.float32)
    nim = minmax_normalize(im, amin, amax, copy)

    assert nim.dtype == "float32"
    assert nim.shape == im.shape
    assert np.logical_and(nim >= 0.0, nim <= 1.0).all()


@pytest.mark.parametrize("norm", [True, False])
def test_float2ubyte(img_sample, norm) -> None:
    im = img_sample.astype(np.float32)
    nim = normalize(im)
    ubyte = float2ubyte(nim, norm)

    assert ubyte.dtype == "uint8"
    assert ubyte.shape == im.shape
    assert np.logical_and(ubyte >= 0, ubyte <= 255).all()
