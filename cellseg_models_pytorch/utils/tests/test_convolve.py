import pytest
import torch

from cellseg_models_pytorch.utils.convolve import filter2D, gaussian_kernel2d


@pytest.mark.parametrize("window_size", [5, 6])
def test_gaussian_kernel2d(window_size):
    kernel = gaussian_kernel2d(window_size, 2)
    assert kernel.shape == torch.Size([1, 1, window_size, window_size])


def test_convolve():
    im = torch.rand([1, 1, 14, 14])
    convolved = filter2D(im, gaussian_kernel2d(5, 2))
    assert convolved.shape == im.shape
