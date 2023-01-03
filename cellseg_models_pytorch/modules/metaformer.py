from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from .base_modules import Identity
from .misc_modules import LayerScale
from .mlp import MlpBlock
from .patch_embeddings import ContiguousEmbed
from .token_mixers import RESHAPE_LOOKUP, TokenMixerBlock


class MetaFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_kwargs: Dict[str, Any],
        mixer_kwargs: Dict[str, Any],
        mlp_kwargs: Dict[str, Any],
        out_channels: int = None,
        layer_scale: bool = False,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        """Create a generic Metaformer block with any token-mixer available.

        Input shape: (B, in_channels, H, W)
        Output shape: (B, out_channels, H, W)

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            embed_kwargs : Dict[str, Any]
                Key-word arguments for the patch embedding block.
            mixer_kwargs : Dict[str, Any]
                Key-word arguments for the token-mixer block.
            mlp_kwargs : Dict[str, Any]
                Key-word arguments for the final Mlp-block.
            out_channels : int, optional
                Number of output channels.
            layer_scale : bool, default=False
                Flag, whether to use layer-scaling.
            dropout : float, default=0.0
                Drop-path probaility.

        Examples
        --------
        MetaFormer with exact memory-efficient self-attention:
        >>> import torch
        >>> import torch.nn as nn

        >>> in_channels = 3
        >>> head_dim = 64
        >>> num_heads = 8
        >>> query_dim = head_dim*num_heads

        >>> # patch embedding kwargs
        >>> embed_kwargs = {
                "in_channels": 3,
                "kernel_size": 7,
                "stride": 4,
                "pad": 2,
                "head_dim": head_dim,
                "num_heads": num_heads,
            }

        >>> # token-mixer kwargs
        >>> mixer_kwargs = {
                "token_mixer": "self-attention",
                "normalization": "ln",
                "residual": True,
                "norm_kwargs": {
                    "normalized_shape": query_dim
                },
                "mixer_kwargs": {
                    "query_dim": query_dim,
                    "name": "exact",
                    "how": "memeff",
                    "cross_attention_dim": None,
                }
            }

        >>> # mlp-kwargs
        >>> mlp_kwargs = {
                "in_channels": query_dim,
                "norm_kwargs": {"normalized_shape": query_dim}
            }

        >>> # init metaformer
        >>> metaformer = MetaFormer(
                in_channels=in_channels,
                embed_kwargs=embed_kwargs,
                mixer_kwargs=mixer_kwargs,
                mlp_kwargs=mlp_kwargs,
                layer_scale=True,
                dropout=0.1
            )

        >>> x = torch.rand([8, 3, 256, 256])
        >>> print(metaformer(x).shape)
        >>> # torch.Size([8, 4096, 512])


        MetaFormer with multi-scale convolutional attention.:
        >>> import torch
        >>> import torch.nn as nn

        >>> in_channels = 3
        >>> head_dim = 64
        >>> num_heads = 8
        >>> query_dim = head_dim*num_heads
        >>> out_channels = 128

        >>> # patch embedding kwargs
        >>> embed_kwargs = {
                "in_channels": 3,
                "kernel_size": 7,
                "stride": 4,
                "pad": 2,
                "head_dim": head_dim,
                "num_heads": num_heads,
            }

        >>> # token-mixer kwargs
        >>> mixer_kwargs = {
            "token_mixer": "mscan",
            "normalization": "bn",
            "norm_kwargs": {
                "num_features": query_dim,
            },
            "mixer_kwargs":{
                "in_channels": query_dim,
            }
        }

        >>> # mlp-kwargs
        >>> mlp_kwargs = {
                "in_channels": query_dim,
                "norm_kwargs": {"normalized_shape": query_dim}
            }

        >>> # init metaformer
        >>> metaformer = MetaFormer(
                in_channels=in_channels,
                out_channels=out_channels,
                embed_kwargs=embed_kwargs,
                mixer_kwargs=mixer_kwargs,
                mlp_kwargs=mlp_kwargs,
                layer_scale=True,
                dropout=0.1
            )

        >>> x = torch.rand([8, 3, 256, 256])
        >>> print(metaformer(x).shape)
        >>> # torch.Size([8, 128, 256, 256])
        """
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        mixer_name = mixer_kwargs["token_mixer"]

        self.patch_embed = ContiguousEmbed(
            **embed_kwargs, flatten=not RESHAPE_LOOKUP[mixer_name]
        )
        self.proj_dim = self.patch_embed.proj_dim

        self.mixer = TokenMixerBlock(**mixer_kwargs)
        self.mlp = MlpBlock(**mlp_kwargs)
        self.ls1 = (
            LayerScale(dim=mlp_kwargs["in_channels"]) if layer_scale else Identity()
        )
        self.ls2 = (
            LayerScale(dim=mlp_kwargs["in_channels"]) if layer_scale else Identity()
        )
        self.drop_path1 = DropPath() if dropout else Identity()
        self.drop_path2 = DropPath() if dropout else Identity()

        self.proj_out = nn.Conv2d(
            self.proj_dim, self.out_channels, kernel_size=1, stride=1, padding=0
        )

        self.downsample = Identity()
        if self.out_channels is not None:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the token mixer module."""
        B, _, H, W = x.shape
        residual = self.downsample(x)

        # 1. embed and project
        x = self.patch_embed(x)

        # 2. token-mixing
        x = self.drop_path1(self.ls1(self.mixer(x, **kwargs)))

        # 3. mlp
        x = self.drop_path2(self.ls2(self.mlp(x)))

        # 4. Reshape back to image-like shape.
        p_H = self.patch_embed.get_patch_size(H)
        p_W = self.patch_embed.get_patch_size(W)
        x = x.reshape(B, p_H, p_W, self.proj_dim).permute(0, 3, 1, 2)

        # Upsample to input dims if patch size less than orig inp size
        # assumes that the input is square mat.
        # NOTE: the kernel_size, pad, & stride has to be set correctly for this to work
        if p_H < H:
            scale_factor = H // p_H
            x = F.interpolate(x, scale_factor=int(scale_factor), mode="bilinear")

        # 5. project to original input channels
        x = self.proj_out(x)

        # 6. residual
        return x + residual
