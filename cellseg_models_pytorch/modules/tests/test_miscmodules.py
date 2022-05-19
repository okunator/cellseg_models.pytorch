import pytest
import torch
import torch.nn as nn

from cellseg_models_pytorch.modules.misc_modules import (
    ChannelPool,
    StyleBlock,
    StyleReshape,
)


@pytest.mark.parametrize("in_channels", [32, 16])
@pytest.mark.parametrize("out_channels", [16, 32])
def test_chpool_fwdbwd(in_channels, out_channels):
    x = torch.rand([1, in_channels, 16, 16])
    chpool = ChannelPool(in_channels, out_channels)
    out = chpool(x)

    out.mean().backward()

    assert out.shape[1] == out_channels


@pytest.mark.parametrize("in_channels", [32])
@pytest.mark.parametrize("style_channels", [32, 16])
@pytest.mark.parametrize("out_channels", [16, 32])
def test_stylefwdbwd(in_channels, style_channels, out_channels):
    x = torch.rand([1, in_channels, 16, 16])
    make_style = StyleReshape(in_channels, style_channels)
    style_block = StyleBlock(style_channels, out_channels)
    conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    style = make_style(x)
    x = conv(x)
    out = style_block(x, style)

    out.mean().backward()

    assert out.shape[1] == out_channels
