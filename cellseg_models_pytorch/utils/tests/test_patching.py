import numpy as np
import pytest

from cellseg_models_pytorch.utils import TilerStitcher, extract_patches, stitch_patches


@pytest.mark.parametrize(
    "patch_shape,multichannel",
    [
        ((256, 256, 3), True),
        ((320, 320, 1), False),
        pytest.param((256, 256), False, marks=pytest.mark.xfail),
        pytest.param((256, 256, 3), False, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize("padding", [True, False])
def test_extract_patches(img_sample, patch_shape, multichannel, padding):
    im = img_sample

    if not multichannel:
        im = im[..., 0]

    patches, _, _ = extract_patches(
        im, stride=80, patch_shape=patch_shape, padding=padding
    )

    assert patches.shape[1:] == patch_shape


def test_stitch_patches(img_sample):
    expected = img_sample

    patch_shape = (128, 128, 3)
    patches, ny, nx = extract_patches(
        expected, stride=80, patch_shape=patch_shape, padding=True
    )

    observed = stitch_patches(patches, expected.shape, ny, nx, 80, padding=True)

    assert expected.shape == observed.shape
    np.testing.assert_array_equal(expected, observed)


@pytest.mark.parametrize(
    "patch_shape,multichannel",
    [
        ((256, 256, 3), True),
        ((320, 320, 1), False),
    ],
)
def test_tilerstitcher(img_sample, patch_shape, multichannel):
    expected = img_sample

    if not multichannel:
        expected = expected[..., 0]

    tiler = TilerStitcher(
        im_shape=expected.shape, patch_shape=patch_shape, stride=256, padding=True
    )

    patches = tiler.patch(expected)
    observed = tiler.backstitch(patches)

    assert expected.shape == observed.shape
    np.testing.assert_array_equal(expected, observed)
