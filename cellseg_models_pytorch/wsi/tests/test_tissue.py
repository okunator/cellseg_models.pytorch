import numpy as np
import pytest

from cellseg_models_pytorch.wsi.tissue import _otsu_threshold, get_tissue_mask


@pytest.mark.parametrize(
    ("pixels", "expected"),
    [
        ([16, 16, 200, 200], 16),
        ([0, 0, 100, 100, 120, 120, 255, 255], 100),
        ([100, 100, 100, 100], 0),
        ([0, 255, 0, 255], 0),
        ([0, 0, 0, 0], 0),
        ([255, 255, 255, 255], 0),
    ],
)
def test_otsu_preserves_threshold_mask_and_input(
    pixels: list[int], expected: int
) -> None:
    image = np.array([pixels], dtype=np.uint8)
    original = image.copy()

    otsu_threshold = _otsu_threshold(gray=image)
    threshold, mask = get_tissue_mask(image, sigma=0)

    assert isinstance(otsu_threshold, float)
    assert otsu_threshold == expected
    assert isinstance(threshold, int)
    assert threshold == expected
    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, (original <= expected).astype(np.uint8))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize(
    ("multiplier", "expected"),
    [(-1.0, 0), (0.0, 0), (0.5, 3), (1.0, 5), (1.5, 8), (100.0, 255)],
)
def test_otsu_multiplier_rounding_and_clipping(
    rgb: bool, multiplier: float, expected: int
) -> None:
    gray = np.array([[0, 5, 5, 200, 255]], dtype=np.uint8)
    image = np.repeat(gray[..., None], 3, axis=-1) if rgb else gray.copy()
    original = image.copy()

    threshold, mask = get_tissue_mask(image, multiplier=multiplier, sigma=0)

    assert isinstance(threshold, int)
    assert threshold == expected
    assert mask.shape == gray.shape
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, (gray <= expected).astype(np.uint8))
    np.testing.assert_array_equal(image, original)


def test_otsu_preserves_default_blurred_threshold_and_mask() -> None:
    image = np.tile(np.array([16, 16, 200, 200], dtype=np.uint8), (4, 1))

    threshold, mask = get_tissue_mask(image)

    assert isinstance(threshold, int)
    assert threshold == 71
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, [[1, 1, 0, 0]] * 4)


@pytest.mark.parametrize("multiplier", [-1.0, 100.0])
def test_explicit_threshold_preserves_integer_result_and_ignores_multiplier(
    multiplier: float,
) -> None:
    image = np.array([[0, 16, 80, 200, 255]], dtype=np.uint8)

    threshold, mask = get_tissue_mask(
        image, threshold=80, multiplier=multiplier, sigma=0
    )

    assert isinstance(threshold, int)
    assert threshold == 80
    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(mask, [[1, 1, 1, 0, 0]])
