import os
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.encoders._base import BaseTrEncoder
from cellseg_models_pytorch.encoders.dinov2.layers import (
    Block,
    MemEffAttention,
    PatchEmbed,
)
from cellseg_models_pytorch.encoders.dinov2.vision_transformer import (
    DinoVisionTransformer,
)

__all__ = ["DinoVit", "build_dinov2_encoder"]


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind  # noqa

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


# name to pre-trained sam weights mapping
MODEL_URLS = {
    "dinov2_vit_small": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",  # noqa
    "dinov2_vit_base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",  # noqa
    "dinov2_vit_large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # noqa
    "dinov2_vit_giant": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",  # noqa
}


class DinoVit(BaseTrEncoder):
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        out_indices: Optional[Tuple[int, ...]] = None,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        qkv_bias: bool = True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        **kwargs,
    ) -> None:
        """Create a wrapper for the DINOv2 backbone.

        Parameters
        ----------
        checkpoint_path : Optional[Union[str, Path]], optional
            Path to the weights of the backbone. If None and pretrained is False the
            backbone is initialized randomly. Defaults to None.
        out_indices : Optional[Tuple[int, ...]], optional
            Indices of the intermediate features to return. If None, only the last
            features are returned. Defaults to None.
        img_size : int, default=224
            Input image size.
        patch_size : int, default=16
            Patch size.
        in_chans : int, default=3
            Number of input image channels.
        embed_dim : int, default=768
            Patch embedding dimension.
        depth : int, default=12
            Depth of ViT.
        num_heads : int, default=12
            Number of attention heads in each ViT block.
        mlp_ratio : float, default=4.0
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias : bool, default=True
            If True, add a learnable bias to query, key, value.
        proj_bias : bool
            Enable bias for proj in attn if True
        ffn_bias : bool
            Enable bias for ffn if True
        drop_path_rate : float
            Stochastic depth rate
        drop_path_uniform : bool
            Apply uniform drop rate across blocks
        weight_init: str
            Weight init scheme
        init_values : float
            Layer-scale init values
        embed_layer : nn.Module
            patch embedding layer
        act_layer : nn.Module
            MLP activation layer
        block_fn : nn.Module
            transformer block class
        ffn_layer : str
            "mlp", "swiglu", "swiglufused" or "identity"
        block_chunks: int
            Split block sequence into block_chunks units for FSDP wrap
        """
        super().__init__(
            name="DINOv2-ViT",
            checkpoint_path=checkpoint_path,
            out_indices=out_indices,
        )
        # self.patch_size = patch_size
        self.patch_size = 16
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_channels = in_chans

        self.backbone = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            ffn_bias=ffn_bias,
            proj_bias=proj_bias,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            init_values=init_values,
            embed_layer=embed_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            ffn_layer=ffn_layer,
            block_chunks=block_chunks,
        )

        if checkpoint_path is not None:
            self.load_checkpoint()

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the backbone and return the selected features."""
        # take the last output as well
        out_inds = list(self.out_indices) + [len(self.backbone.blocks) - 1]
        out_inds = sorted(set(out_inds))
        feats = self.backbone.get_intermediate_layers(
            x, out_inds, reshape=True, return_class_token=True, norm=True
        )

        features, cls_tokens = list(zip(*feats))
        output = self.backbone.head(features[-1][:, 0])

        return torch.mean(output, axis=-1), output, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the backbone."""
        logits, cls_token, features = self.forward_features(x)

        return features


def build_dinov2_encoder(
    name: str, pretrained: bool = True, checkpoint_path: str = None, **kwargs
) -> nn.Module:
    """Build a DINOv2 ViT encoder.

    Parameters
    ----------
    name : str
        Name of the encoder. Must be one of "dinov2_vit_small", "dinov2_vit_base",
        "dinov2_vit_large", "dinov2_vit_giant".
    pretrained : bool, optional
        If True, load pretrained weights, by default True.
    checkpoint_path : str, optional
        Path to the weights of the backbone. If None, the backbone is initialized
        randomly. Defaults to None.
    **kwargs
        Arbitrary key-word arguments for the `DinoVit`.

    Returns
    -------
    nn.Module
        The initialized DINOv2 ViT encoder.
    """
    allowed = (
        "dinov2_vit_small",
        "dinov2_vit_base",
        "dinov2_vit_large",
        "dinov2_vit_giant",
    )
    if name not in allowed:
        raise ValueError(f"Unknown encoder name: {name}, allowed values are {allowed}")

    if checkpoint_path is None and pretrained:
        checkpoint_path = MODEL_URLS[name]

    if kwargs.get("block_fn", None) is None:
        block_fn = (
            partial(Block, attn_class=MemEffAttention) if XFORMERS_AVAILABLE else Block
        )
    else:
        block_fn = kwargs.pop("block_fn")

    if name == "dinov2_vit_small":
        dino_vit = DinoVit(
            checkpoint_path=checkpoint_path,
            out_indices=[3, 6, 9, 11],
            img_size=518,
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            block_chunks=0,
            block_fn=block_fn,
            init_values=1.0,
            ffn_layer="mlp",
            **kwargs,
        )
    elif name == "dinov2_vit_base":
        dino_vit = DinoVit(
            checkpoint_path=checkpoint_path,
            out_indices=[3, 6, 9, 11],
            img_size=518,
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            block_chunks=0,
            block_fn=block_fn,
            init_values=1.0,
            ffn_layer="mlp",
            **kwargs,
        )
    elif name == "dinov2_vit_large":
        dino_vit = DinoVit(
            checkpoint_path=checkpoint_path,
            out_indices=[6, 12, 18, 23],
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            block_chunks=0,
            block_fn=block_fn,
            init_values=1.0,
            ffn_layer="mlp",
            **kwargs,
        )
    elif name == "dinov2_vit_giant":
        dino_vit = DinoVit(
            checkpoint_path=checkpoint_path,
            out_indices=[10, 22, 32, 39],
            img_size=518,
            patch_size=14,
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=4,
            block_chunks=0,
            block_fn=block_fn,
            init_values=1.0,
            ffn_layer="mlp",
            **kwargs,
        )

    return dino_vit
