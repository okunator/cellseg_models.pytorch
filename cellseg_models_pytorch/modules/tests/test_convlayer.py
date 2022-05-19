import pytest
import torch

from cellseg_models_pytorch.modules import ConvLayer


@pytest.mark.parametrize("convblocks", [("basic", "dws"), ("mbconv", "fmbconv")])
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
def test_conv_block_forward(convblocks, short_skip):
    input = torch.rand([1, 8, 16, 16])
    layer = ConvLayer(
        in_channels=8,
        out_channels=4,
        block_types=convblocks,
        short_skip=short_skip,
        attentions=(None, "gc"),
        expand_ratio=1.0,
    )

    output = layer(input)

    assert output.shape == torch.Size([1, 4, 16, 16])
    assert output.dtype == input.dtype


@pytest.mark.parametrize("convblocks", [("basic", "dws"), ("mbconv", "fmbconv")])
@pytest.mark.parametrize("short_skip", ["residual", "dense", "basic"])
def test_conv_block_backward(convblocks, short_skip):
    input = torch.rand([1, 8, 16, 16])
    layer = ConvLayer(
        in_channels=8,
        out_channels=4,
        block_types=convblocks,
        short_skip=short_skip,
        attentions=(None, "gc"),
        expand_ratio=1.0,
    )

    output = layer(input)
    output.mean().backward()

    assert output.shape == torch.Size([1, 4, 16, 16])
    assert output.dtype == input.dtype
