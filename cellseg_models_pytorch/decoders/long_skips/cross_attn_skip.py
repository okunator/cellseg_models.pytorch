from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...modules import Transformer2D
from ...modules.patch_embeddings import ContiguousEmbed

__all__ = ["CrossAttentionSkip"]


class CrossAttentionSkip(nn.Module):
    def __init__(
        self,
        stage_ix: int,
        in_channels: int,
        skip_channels: Tuple[int, ...] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        n_blocks: int = 2,
        block_types: Tuple[str, ...] = ("exact", "exact"),
        computation_types: Tuple[str, ...] = ("basic", "basic"),
        dropouts: Tuple[float, ...] = (0.0, 0.0),
        biases: Tuple[bool, ...] = (False, False),
        layer_scales: Tuple[bool, ...] = (False, False),
        activation: str = "star_relu",
        mlp_ratio: int = 2,
        slice_size: int = 4,
        patch_embed_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Skip connection (U-Net-like) via cross-attention.

        Does the long skip connection through a cross-attention transformer rather than
        merging or summing the skip features to the upsampled decoder feature-map.

        Args:
            stage_ix: Index number signalling the current decoder stage
            in_channels: The number of channels in the input tensor.
                Defaults to None.
            skip_channels: Tuple of the number of channels in the encoder stages.
                Order is bottom up. This list does not include the final
                bottleneck stage out channels.
            num_heads: Number of heads in multi-head attention.
                Defaults to 8.
            head_dim: The out dim of the heads.
                Defaults to 64.
            n_blocks: Number of SelfAttentionBlocks used in this layer.
                Defaults to 1.
            block_types:
                The name of the SelfAttentionBlocks in the TransformerLayer.
                Length of the tuple has to equal `n_blocks`
                Allowed names: "basic". "slice", "flash".
                Defaults to ("exact", ).
            computation_types:
                The way of computing the attention matrices in the SelfAttentionBlocks
                in the TransformerLayer. Length of the tuple has to equal `n_blocks`
                Allowed styles: "basic". "slice", "flash", "memeff", "slice_memeff".
                Defaults to ("basic", ).
            dropouts: Dropout probabilities for the SelfAttention blocks.
                Defaults to (False, ).
            biases (bool): Include bias terms in the SelfAttention blocks.
                Defaults to (True, True).
            layer_scales (bool):
                Learnable layer weights for the self-attention matrix.
                Defaults to (False, ).
            activation:
                The activation function applied at the end of the transformer layer fc.
                One of ("geglu", "approximate_gelu", "star_relu").
                Defaults to "star_relu".
            mlp_ratio:
                Multiplier that defines the out dimension of the final fc projection
                layer.
                Defaults to 4.
            slice_size: Slice size for sliced self-attention. This is used only if
                `name = "slice"` for a SelfAttentionBlock.
                Defaults to 4.
            patch_embed_kwargs (Dict[str, Any]):
                Extra key-word arguments for the patch embedding module. See the
                `ContiguousEmbed` module for more info.
        """
        super().__init__()
        self.in_channels = in_channels
        self.stage_ix = stage_ix

        if stage_ix < len(skip_channels):
            context_channels = skip_channels[stage_ix]

            self.context_patch_embed = ContiguousEmbed(
                in_channels=context_channels,
                patch_size=1,
                num_heads=num_heads,
                head_dim=head_dim,
                normalization="gn",
                norm_kwargs={"num_features": context_channels},
                **patch_embed_kwargs if patch_embed_kwargs is not None else {},
            )

            self.tranformer = Transformer2D(
                in_channels=in_channels,
                cross_attentions_dims=self.context_patch_embed.proj_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                n_blocks=n_blocks,
                block_types=block_types,
                computation_types=computation_types,
                dropouts=dropouts,
                biases=biases,
                layer_scales=layer_scales,
                activation=activation,
                slice_size=slice_size,
                mlp_ratio=mlp_ratio,
                patch_embed_kwargs=patch_embed_kwargs,
                **kwargs,
            )

    @property
    def out_channels(self) -> int:
        """Out channels."""
        return self.in_channels

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """Forward pass of the skip connection."""
        if self.stage_ix < len(skips):
            context = skips[self.stage_ix]  # (B, C, H, W)

            # embed context for cross-attm transformer: (B, H'*W', num_heads*head_dim)
            context = self.context_patch_embed(context)

            x = self.tranformer(x, context=context)  # (B, C, H, W)

        return x
