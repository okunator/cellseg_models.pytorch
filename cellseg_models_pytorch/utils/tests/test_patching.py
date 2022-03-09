import numpy as np
import pytest
import torch

from cellseg_models_pytorch.utils import (
    TilerStitcher,
    TilerStitcherTorch,
    extract_patches_numpy,
    extract_patches_torch,
    stitch_patches_numpy,
    stitch_patches_torch,
)


@pytest.fixture
def rand_tensor():
    return torch.rand([2, 3, 410, 520])


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
def test_extract_patches_numpy(img_sample, patch_shape, multichannel, padding):
    im = img_sample

    if not multichannel:
        im = im[..., 0]

    patches, _, _ = extract_patches_numpy(
        im, stride=80, patch_shape=patch_shape, padding=padding
    )

    assert patches.shape[1:] == patch_shape


def test_stitch_patches_numpy(img_sample):
    expected = img_sample

    patch_shape = (128, 128, 3)
    patches, ny, nx = extract_patches_numpy(
        expected, stride=80, patch_shape=patch_shape, padding=True
    )

    observed = stitch_patches_numpy(patches, expected.shape, ny, nx, 80, padding=True)

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


@pytest.mark.parametrize("padding", [True, False])
def test_extract_patches_torch(rand_tensor, padding):
    out = extract_patches_torch(rand_tensor, 256, (256, 256), padding)

    assert out.shape[-2:] == torch.Size([256, 256])


@pytest.mark.parametrize("padding", [True, False])
def test_stitch_patches_torch(rand_tensor, padding):
    stride = 256
    patches = extract_patches_torch(rand_tensor, stride, (256, 256), padding)
    stitched = stitch_patches_torch(patches, tuple(rand_tensor.shape), stride, padding)

    assert stitched.shape == rand_tensor.shape


@pytest.mark.parametrize("padding", [True, False])
def test_tilerstitchertorch(rand_tensor, padding):
    ts = TilerStitcherTorch(tuple(rand_tensor.shape), (256, 256), 256, padding)

    patches = ts.patch(rand_tensor)
    stitched = ts.backstitch(patches)

    assert stitched.shape == rand_tensor.shape
