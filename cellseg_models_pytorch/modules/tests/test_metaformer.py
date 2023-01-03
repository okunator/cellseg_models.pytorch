import pytest
import torch

from cellseg_models_pytorch.modules import MetaFormer


@pytest.mark.parametrize("type", ["pool", "mscan", "mlp", "self-attention"])
def test_metaformer(type):
    in_channels = 3
    head_dim = 64
    num_heads = 8
    query_dim = head_dim * num_heads
    out_channels = 16

    embed_kwargs = {
        "in_channels": 3,
        "kernel_size": 7,
        "stride": 4,
        "pad": 2,
        "head_dim": head_dim,
        "num_heads": num_heads,
    }

    if type == "self-attention":
        mixer_kwargs = {
            "token_mixer": "self-attention",
            "normalization": "ln",
            "residual": True,
            "norm_kwargs": {"normalized_shape": query_dim},
            "mixer_kwargs": {
                "query_dim": query_dim,
                "name": "exact",
                "how": "basic",
                "cross_attention_dim": None,
            },
        }

    elif type == "mlp":
        mixer_kwargs = {
            "token_mixer": "mlp",
            "normalization": "ln",
            "norm_kwargs": {"normalized_shape": query_dim},
            "mixer_kwargs": {
                "in_channels": query_dim,
            },
        }

    elif type in ("pool", "mscan"):
        mixer_kwargs = {
            "token_mixer": type,
            "normalization": "bn",
            "norm_kwargs": {
                "num_features": query_dim,
            },
            "mixer_kwargs": {
                "kernel_size": 3,
                "in_channels": query_dim,
            },
        }

    mlp_kwargs = {
        "in_channels": query_dim,
        "norm_kwargs": {"normalized_shape": query_dim},
    }

    metaformer = MetaFormer(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_kwargs=embed_kwargs,
        mixer_kwargs=mixer_kwargs,
        mlp_kwargs=mlp_kwargs,
        layer_scale=True,
        dropout=0.1,
    )

    x = torch.rand([8, 3, 32, 32])
    dd = metaformer(x)

    assert dd.shape == torch.Size([8, out_channels, 32, 32])
