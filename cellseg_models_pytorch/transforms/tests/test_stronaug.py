import numpy as np
import pytest
from PIL import Image, ImageOps

from cellseg_models_pytorch.transforms.functional.generic_transforms import (
    AUGMENT_SPACE,
    _apply_operation,
    _magnitude_kwargs,
    posterize,
)


@pytest.mark.parametrize("bits", range(9))
def test_posterize_matches_pillow(bits: int) -> None:
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = np.stack((gray, np.rot90(gray), np.flipud(gray)), axis=-1)
    original = image.copy()
    expected = np.asarray(ImageOps.posterize(Image.fromarray(image), bits))

    actual = posterize(image, bits)

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("op_name", list(AUGMENT_SPACE.keys()))
def test_stronaug_transforms(img_sample, op_name):
    rng = np.random.RandomState(seed=123)
    kwargs = dict(
        name=op_name,
        **_magnitude_kwargs(op_name, bounds=AUGMENT_SPACE[op_name], rng=rng)
    )

    tr_img = _apply_operation(img_sample, op_name, **kwargs)

    assert tr_img.shape == img_sample.shape
    assert tr_img.dtype == img_sample.dtype
    assert tr_img.max() <= 255
