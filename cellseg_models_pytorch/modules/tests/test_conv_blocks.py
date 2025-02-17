import pytest
import torch

from cellseg_models_pytorch.modules import ConvBlock
from cellseg_models_pytorch.modules.mlp import ConvMlp


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


@pytest.mark.parametrize("in_channels, out_channels, input_shape", [
    (32, 16, (1, 32, 32, 32)),
    (32, None, (2, 32, 32, 32)),
])
def test_convmlp(in_channels, out_channels, input_shape):
    conv_mlp = ConvMlp(
        in_channels=in_channels,
        out_channels=out_channels,
        mlp_ratio=1
    )

    if out_channels is None:
        out_channels = in_channels

    # Create a random input tensor with the specified shape
    x = torch.rand(input_shape).float()

    # Forward pass
    output = conv_mlp(x)

    # Check the output shape
    expected_out_channels = in_channels if out_channels is None else out_channels
    assert output.shape == (input_shape[0], expected_out_channels, input_shape[2], input_shape[3])

    # Check the output type
    assert isinstance(output, torch.Tensor)