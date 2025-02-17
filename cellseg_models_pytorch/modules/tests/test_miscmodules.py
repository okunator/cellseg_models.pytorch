import pytest
import torch
import torch.nn as nn

from cellseg_models_pytorch.modules.misc_modules import (
    ChannelPool,
    StyleBlock,
    StyleReshape,
)
from cellseg_models_pytorch.modules.patch_embeddings import PatchEmbed

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


@pytest.mark.parametrize("in_channels, patch_size, head_dim, num_heads, input_shape", [
    (3, 8, 32, 4, (1, 3, 128, 128)),
    (3, 4, 16, 2, (1, 3, 64, 64)),
])
def test_patch_embed(in_channels, patch_size, head_dim, num_heads, input_shape):
    # Create the PatchEmbed layer
    patch_embed = PatchEmbed(
        in_channels=in_channels,
        patch_size=patch_size,
        head_dim=head_dim,
        num_heads=num_heads
    )

    # Create a random input tensor with the specified shape
    x = torch.randn(input_shape)

    # Forward pass
    output = patch_embed(x)

    # Calculate expected output shape
    B, _, H, W = input_shape
    expected_seq_len = (H // patch_size) * (W // patch_size)
    expected_proj_dim = head_dim * num_heads

    # Check the output shape
    assert output.shape == (B, expected_seq_len, expected_proj_dim)

    # Check the output type
    assert isinstance(output, torch.Tensor)
