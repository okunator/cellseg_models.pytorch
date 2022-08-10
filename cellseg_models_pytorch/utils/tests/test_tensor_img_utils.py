import pytest
import torch

from cellseg_models_pytorch.utils import (
    dataset_normalize_torch,
    minmax_normalize_torch,
    ndarray_to_tensor,
    normalize_torch,
    percentile_normalize_torch,
)


@pytest.mark.parametrize(
    "norm_method", [minmax_normalize_torch, percentile_normalize_torch, normalize_torch]
)
@pytest.mark.parametrize("batch_dim", [True, False])
def test_channel_norms(img_sample, norm_method, batch_dim):

    if batch_dim:
        img = ndarray_to_tensor(img_sample, "HWC", "BCHW")
    else:
        img = ndarray_to_tensor(img_sample, "HWC", "HWC").permute(2, 0, 1)

    img = norm_method(img)

    assert img.dtype == torch.float32
    assert img.shape == img.shape


@pytest.mark.parametrize("batch_dim", [True, False])
def test_dataset_norm(img_sample, batch_dim):
    if batch_dim:
        img = ndarray_to_tensor(img_sample, "HWC", "BCHW")
    else:
        img = ndarray_to_tensor(img_sample, "HWC", "HWC").permute(2, 0, 1)

    img = dataset_normalize_torch(img, (0.4, 0.5, 0.6), (0.1, 0.2, 0.1), True)

    assert img.dtype == torch.float32
    assert img.shape == img.shape
