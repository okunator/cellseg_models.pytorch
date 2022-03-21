import pytest
import torch

from cellseg_models_pytorch.modules import ConvBlock


@pytest.mark.parametrize(
    "convblock", ["basic", "mbconv", "fmbconv", "dws", "bottleneck"]
)
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
@pytest.mark.parametrize("preattend", [True, False])
@pytest.mark.parametrize("preactivate", [True, False])
@pytest.mark.parametrize("out_channels", [4, 8])
def test_conv_block_forward(
    convblock, short_skip, preattend, preactivate, out_channels
):
    input = torch.rand([1, 8, 16, 16])
    block = ConvBlock(
        name=convblock,
        short_skip=short_skip,
        in_channels=8,
        out_channels=out_channels,
        attention="gc",
        preactivate=preactivate,
        preattend=preattend,
        expand_ratio=1.0,
    )

    output = block(input)

    assert output.shape == torch.Size([1, out_channels, 16, 16])
    assert output.dtype == input.dtype


@pytest.mark.parametrize(
    "convblock", ["basic", "mbconv", "fmbconv", "dws", "bottleneck"]
)
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
def test_conv_block_backward(convblock, short_skip):
    input = torch.rand([1, 8, 16, 16])
    block = ConvBlock(
        name=convblock,
        short_skip=short_skip,
        in_channels=8,
        out_channels=4,
        attention="gc",
        preactivate=False,
        preattend=False,
        expand_ratio=1.0,
    )

    output = block(input)
    output.mean().backward()

    assert output.shape == torch.Size([1, 4, 16, 16])
    assert output.dtype == input.dtype
