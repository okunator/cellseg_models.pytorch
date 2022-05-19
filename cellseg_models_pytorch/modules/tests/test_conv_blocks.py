import pytest
import torch

from cellseg_models_pytorch.modules import ConvBlock


@pytest.mark.parametrize(
    "convblock", ["basic", "mbconv", "fmbconv", "dws", "bottleneck", "hover_dense"]
)
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
@pytest.mark.parametrize("preattend", [True, False])
@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("style_channels", [None, 4])
@pytest.mark.parametrize("out_channels", [4, 8])
def test_conv_block_fwdbwd(
    convblock,
    short_skip,
    preattend,
    preactivate,
    out_channels,
    style_channels,
):
    input = torch.rand([1, 8, 16, 16])

    style = None
    if style_channels is not None:
        style = torch.ones([1, style_channels])

    block = ConvBlock(
        name=convblock,
        short_skip=short_skip,
        in_channels=8,
        out_channels=out_channels,
        style_channels=style_channels,
        attention="gc",
        preactivate=preactivate,
        preattend=preattend,
        expand_ratio=1.0,
    )

    output = block(input, style)
    output.mean().backward()

    assert output.shape == torch.Size([1, out_channels, 16, 16])
    assert output.dtype == input.dtype
