import pytest
import torch

from cellseg_models_pytorch.utils import filter2D, gaussian_kernel2d, sobel_hv


@pytest.mark.parametrize("window_size", [5, 6])
def test_gaussian_kernel2d(window_size):
    kernel = gaussian_kernel2d(window_size, 2)
    assert kernel.shape == torch.Size([1, 1, window_size, window_size])


@pytest.mark.parametrize("window_size", [5, pytest.param(6, marks=pytest.mark.xfail)])
def test_sobel_kernel(window_size):
    kernel = sobel_hv(window_size)
    assert kernel.shape == torch.Size([2, 1, window_size, window_size])


def test_convolve():
    im = torch.rand([1, 1, 14, 14])
    convolved = filter2D(im, gaussian_kernel2d(5, 2))
    assert convolved.shape == im.shape
