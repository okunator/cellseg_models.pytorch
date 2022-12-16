import pytest
import torch

from cellseg_models_pytorch.modules import Transformer2D


@pytest.mark.parametrize("block_type", ["basic", "slice"])
def test_transformer(block_type):
    in_channels = 64
    B = 4
    H = W = 32

    x = torch.rand([B, in_channels, H, W])
    tr = Transformer2D(
        in_channels=in_channels,
        num_heads=4,
        head_dim=32,
        n_blocks=1,
        block_types=(block_type,),
        biases=(False,),
        dropouts=(0.0,),
        slice_size=4,
    )

    out = tr(x)

    assert out.shape == x.shape