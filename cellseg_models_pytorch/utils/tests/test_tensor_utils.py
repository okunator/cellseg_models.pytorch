import numpy as np
import pytest
import torch

from cellseg_models_pytorch.utils import (
    ndarray_to_tensor,
    tensor_one_hot,
    tensor_to_ndarray,
    to_device,
)


@pytest.mark.parametrize(
    "in_dim_format",
    ("HW", "HWC", "BHWC", "BCHW", "BHW", pytest.param("B", marks=pytest.mark.xfail)),
)
@pytest.mark.parametrize(
    "out_dim_format",
    ("HW", "HWC", "BHWC", "BCHW", "BHW", pytest.param("B", marks=pytest.mark.xfail)),
)
def test_ndarray_to_tensor(in_dim_format, out_dim_format):
    arr = np.random.rand(14, 14)

    if in_dim_format in ("HWC", "BHWC", "BCHW"):
        arr = arr[..., None]

    if in_dim_format in ("BHWC", "BCHW", "BHW"):
        arr = arr[None, ...]

    if len(arr.shape) == 4 and in_dim_format == "BCHW":
        arr = arr.transpose(0, 3, 1, 2)

    t = ndarray_to_tensor(arr, in_dim_format, out_dim_format)
    assert isinstance(t, torch.Tensor)


@pytest.mark.parametrize(
    "in_dim_format",
    ("BCHW", "BHW", pytest.param("BHWC", marks=pytest.mark.xfail)),
)
@pytest.mark.parametrize(
    "out_dim_format",
    ("BHWC", "BHW", "HWC", "HW", pytest.param("B", marks=pytest.mark.xfail)),
)
def test_tensor_to_ndarray(in_dim_format, out_dim_format):

    if in_dim_format == "BCHW":
        arr = np.random.rand(2, 2, 14, 14)
    else:
        arr = np.random.rand(2, 14, 14)

    t = ndarray_to_tensor(arr, in_dim_format, in_dim_format)
    n = tensor_to_ndarray(t, in_dim_format, out_dim_format)
    assert isinstance(n, np.ndarray)


@pytest.mark.cuda
def test_to_device():
    tensor_cpu = torch.randint(1, 2, [3, 3])
    t = to_device(tensor_cpu)

    assert t.is_cuda


def test_tensor_one_hot(tensor_sem_map):
    t = tensor_one_hot(tensor_sem_map, n_classes=4)
    assert t.shape == torch.Size([2, 4, 6, 6])
    assert t.type() == "torch.FloatTensor"
