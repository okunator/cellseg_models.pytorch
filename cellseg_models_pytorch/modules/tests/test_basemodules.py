import pytest
import torch

from cellseg_models_pytorch.modules.base_modules import Activation, Conv, Norm, Up


@pytest.mark.parametrize("activation", ["leaky-relu", "relu", "mish", "swish", None])
def test_act_forward(activation):
    act = Activation(activation)
    input = torch.rand([1, 3, 16, 16])
    output = act(input)

    assert output.dtype == input.dtype


@pytest.mark.parametrize("normalization", ["bn", "bcn", "gn", "ln2d", None])
def test_norm(normalization):
    norm = Norm(normalization, num_features=3)
    input = torch.rand([1, 3, 16, 16])
    output = norm(input)

    assert output.dtype == input.dtype


@pytest.mark.parametrize("scale_factor", [2, 4])
@pytest.mark.parametrize("upsampling", ["fixed-unpool", "bilinear", "bicubic"])
def test_up(upsampling, scale_factor):
    up = Up(upsampling, scale_factor)
    input = torch.rand([1, 3, 16, 16])
    output = up(input)

    assert output.dtype == input.dtype
    assert output.shape == torch.Size([1, 3, 16 * scale_factor, 16 * scale_factor])


@pytest.mark.parametrize("conv", ["conv", "wsconv", "scaled_wsconv"])
def test_conv(conv):
    conv = Conv(conv, in_channels=3, out_channels=3, kernel_size=3, padding=1)
    input = torch.rand([1, 3, 16, 16])
    output = conv(input)

    assert output.dtype == input.dtype
    assert output.shape == input.shape
