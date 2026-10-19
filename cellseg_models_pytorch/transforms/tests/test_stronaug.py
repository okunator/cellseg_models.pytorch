import numpy as np
import pytest
from PIL import Image, ImageOps

from cellseg_models_pytorch.transforms.functional.generic_transforms import (
    AUGMENT_SPACE,
    _apply_operation,
    _magnitude_kwargs,
    posterize,
)


@pytest.mark.parametrize("bits", range(1, 9))
def test_posterize_matches_pillow(bits: int) -> None:
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    image = np.stack((gray, np.rot90(gray), np.flipud(gray)), axis=-1)
    original = image.copy()
    expected = np.asarray(ImageOps.posterize(Image.fromarray(image), bits))

    actual = posterize(image, bits)

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("rgb", [False, True])
def test_posterize_zero_bits_clears_all_values(rgb: bool) -> None:
    # Pillow 13 rejects zero bits; our public range remains 0–8.
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    if rgb:
        image = np.stack((image, np.rot90(image), np.flipud(image)), axis=-1)
    original = image.copy()

    actual = posterize(image, 0)

    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, np.zeros_like(image))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("op_name", list(AUGMENT_SPACE.keys()))
def test_stronaug_transforms(img_sample, op_name):
    rng = np.random.RandomState(seed=123)
    kwargs = dict(
        name=op_name,
        **_magnitude_kwargs(op_name, bounds=AUGMENT_SPACE[op_name], rng=rng),
    )

    tr_img = _apply_operation(img_sample, op_name, **kwargs)

    assert tr_img.shape == img_sample.shape
    assert tr_img.dtype == img_sample.dtype
    assert tr_img.max() <= 255


def test_strong_augment_composes_selected_operations():
    pytest.importorskip("albumentations")
    from cellseg_models_pytorch.transforms.albu_transforms import AlbuStrongAugment

    image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
    mask = np.arange(12 * 12, dtype=np.int32).reshape(12, 12)
    transform = AlbuStrongAugment(
        augment_space={"brightness": (0.5, 0.5), "gamma": (2.0, 2.0)},
        operations=(2,),
        probabilites=(1.0,),
        seed=0,
    )
    result = transform(image=image.copy(), masks=[mask])
    expected = image.copy()
    for operation, params in transform.last_operations.items():
        expected = _apply_operation(expected, operation, **params)
    np.testing.assert_array_equal(result["image"], expected)
    np.testing.assert_array_equal(result["masks"][0], mask)
    assert len(transform.last_operations) == 2
