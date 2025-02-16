import numpy as np
import pytest
import torch

from cellseg_models_pytorch.utils import to_tensor, to_device

@pytest.mark.parametrize("arr", [np.random.rand(14, 14), np.random.rand(14, 14, 3)])
def test_to_tensor(arr):
    t = to_tensor(arr)
    assert isinstance(t, torch.Tensor)


@pytest.mark.cuda
def test_to_device():
    tensor_cpu = torch.randint(1, 2, [3, 3])
    t = to_device(tensor_cpu)

    assert t.is_cuda
