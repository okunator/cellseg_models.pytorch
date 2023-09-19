import pytest
import torch

from cellseg_models_pytorch.encoders.dino_vit import build_dinov2_encoder
from cellseg_models_pytorch.encoders.dinov2.layers import Block


@pytest.mark.parametrize(
    "name",
    ["dinov2_vit_small", "dinov2_vit_base", "dinov2_vit_large", "dinov2_vit_giant"],
)
def test_dinov2_fwd(name):
    dino = build_dinov2_encoder(name=name, pretrained=False, block_fn=Block)

    x = torch.randn(1, 3, 28, 28)
    feats = dino(x)
    feat_shapes = [f.shape for f in feats]
    expected_shapes = [
        torch.Size([1, dino.embed_dim, 2, 2]),
        torch.Size([1, dino.embed_dim, 2, 2]),
        torch.Size([1, dino.embed_dim, 2, 2]),
        torch.Size([1, dino.embed_dim, 2, 2]),
    ]
    assert feat_shapes == expected_shapes
