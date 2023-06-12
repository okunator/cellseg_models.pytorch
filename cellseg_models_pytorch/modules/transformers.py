from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_modules import Identity
from .misc_modules import LayerScale
from .mlp import MlpBlock
from .patch_embeddings import ContiguousEmbed
from .self_attention_modules import SelfAttentionBlock

__all__ = ["Transformer2D", "TransformerLayer"]


class Transformer2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        cross_attention_dim: int = None,
        n_blocks: int = 2,
        block_types: Tuple[str, ...] = ("exact", "exact"),
        computation_types: Tuple[str, ...] = ("basic", "basic"),
        dropouts: Tuple[float, ...] = (0.0, 0.0),
        biases: Tuple[bool, ...] = (False, False),
        layer_scales: Tuple[bool, ...] = (False, False),
        activation: str = "star_relu",
        num_groups: int = 32,
        mlp_ratio: int = 2,
        slice_size: Optional[int] = 4,
        patch_embed_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Create a transformer for 2D-image-like (B, C, H, W) inputs.

        NOTE: The output shape is the same size as the input shape.

        Parameters
        ----------
            in_channels : int
                Number of channels in the 2D-input of shape (B, in_channels, H, W).
            num_heads : int, default=8
                Number of heads in multi-head attention.
            head_dim : int, default=64
                The out dim of the heads.
            cross_attention_dims : int, optional
                The out dim/length of the context query tensor. Cross attention combines
                asymmetrically two separate embeddings (context and input embeddings).
                E.g. passage from transformer encoder to transformer decoder. If this is
                set to None, no cross attention is applied.
            n_blocks : int, default=2
                Number of Multihead attention blocks in the transformer.
            block_types : Tuple[str, ...], default=("exact", "exact")
                The names/types of the SelfAttentionBlocks in the TransformerLayer.
                Length of the tuple has to equal `n_blocks`.
                Allowed names: ("exact", "linformer").
            computation_types : Tuple[str, ...], default=("basic", "basic")
                The way of computing the attention matrices in the SelfAttentionBlocks
                in the TransformerLayer. Length of the tuple has to equal `n_blocks`
                Allowed styles: "basic". "slice", "flash", "memeff", "slice_memeff".
            dropouts : Tuple[float, ...], default=(False, False)
                Dropout probabilities for the SelfAttention blocks.
            biases : bool, default=(True, True)
                Include bias terms in the SelfAttention blocks.
            layer_scales : bool, default=(False, False)
                Learnable layer weights for the self-attention matrix.
            activation : str, default="star_relu"
                The activation function applied at the end of the transformer layer fc.
                One of ("geglu", "approximate_gelu", "star_relu").
            num_groups : int, default=32
                Number of groups in the first group-norm op before the input is
                projected to be suitable for self-attention.
            mlp_ratio : int, default=2
                Scaling factor for the number of input features to get the number of
                hidden features in the final `Mlp` layer of the transformer.
            slice_size : int, optional, default=4
                Slice size for sliced self-attention. This is used only if
                `name = "slice"` for a SelfAttentionBlock.
            patch_embed_kwargs: Dict[str, Any], optional
                Extra key-word arguments for the patch embedding module. See the
                `ContiguousEmbed` module for more info.
        """
        super().__init__()
        patch_norm = "gn" if in_channels >= 32 else None
        self.patch_embed = ContiguousEmbed(
            in_channels=in_channels,
            patch_size=1,
            head_dim=head_dim,
            num_heads=num_heads,
            normalization=patch_norm,
            norm_kwargs={"num_features": in_channels, "num_groups": num_groups},
            **patch_embed_kwargs if patch_embed_kwargs is not None else {},
        )
        self.proj_dim = self.patch_embed.proj_dim

        self.transformer = TransformerLayer(
            query_dim=self.proj_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
            n_blocks=n_blocks,
            block_types=block_types,
            computation_types=computation_types,
            dropouts=dropouts,
            biases=biases,
            layer_scales=layer_scales,
            activation=activation,
            slice_size=slice_size,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )

        self.proj_out = nn.Conv2d(
            self.proj_dim, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the 2D transformer.

        Parameters
        ----------
            x : torch.Tensor
                Input image-like tensor. Shape (B, C, H, W).
            context : torch.Tensor, optional
                Context tensor. Shape (B, H*W, query_dim)). Can be used to add context
                information from another source than the current transformer.

        Returns
        -------
            torch.Tensor:
                Self-attended input tensor. Same shape as output.
        """
        B, _, H, W = x.shape
        residual = x

        # 1. embed and project
        x = self.patch_embed(x)

        # 2. transformer
        x = self.transformer(x, context)

        # 3. Reshape back to image-like shape.
        p_H = self.patch_embed.get_patch_size(H)
        p_W = self.patch_embed.get_patch_size(W)
        x = x.reshape(B, p_H, p_W, self.proj_dim).permute(0, 3, 1, 2)

        # Upsample to input dims if patch size less than orig inp size
        # assumes that the input is square mat.
        # NOTE: the kernel_size, pad, & stride has to be set correctly for this to work
        if p_H < H:
            scale_factor = H // p_H
            x = F.interpolate(x, scale_factor=int(scale_factor), mode="bilinear")

        # 4. project to original input channels
        x = self.proj_out(x)

        # 5. residual
        return x + residual


class TransformerLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        cross_attention_dim: int = None,
        activation: str = "star_relu",
        n_blocks: int = 2,
        block_types: Tuple[str, ...] = ("exact", "exact"),
        computation_types: Tuple[str, ...] = ("basic", "basic"),
        dropouts: Tuple[float, ...] = (0.0, 0.0),
        biases: Tuple[bool, ...] = (False, False),
        layer_scales: Tuple[bool, ...] = (False, False),
        mlp_ratio: int = 2,
        slice_size: Optional[int] = 4,
        **kwargs,
    ) -> None:
        """Chain transformer blocks to compose a full generic transformer.

        NOTE: For 2D image like data:
            - Forward input shape: (B, H*W, head_dim*num_heads)
            - Forward output sahpe: (B, H*W, head_dim*num_heads)

        Parameters
        ----------
            query_dim : int
                The length/dim of the query. Typically: num_heads*head_dim
            num_heads : int, default=8
                Number of heads in multi-head attention.
            head_dim : int, default=64
                The out dim of the heads.
            cross_attention_dims : int, optional
                The out dim/length of the context query tensor. Cross attention combines
                asymmetrically two separate embeddings (context and input embeddings).
                E.g. passage from transformer encoder to transformer decoder. If this is
                set to None, no cross attention is applied.
            activation : str, default="star_relu"
                The activation function applied at the end of the transformer layer fc.
                One of ("gelu", "geglu", "approximate_gelu", "star_relu").
            n_blocks : int, default=2
                Number of SelfAttentionBlocks used in this layer.
            block_types : Tuple[str, ...], default=("exact", "exact")
                The name/type of the SelfAttentionBlocks in the TransformerLayer.
                Length of the tuple has to equal `n_blocks`.
                Allowed names: ("exact", "linformer").
            computation_types : Tuple[str, ...], default=("basic", "basic")
                The way of computing the attention matrices in the SelfAttentionBlocks
                in the TransformerLayer. Length of the tuple has to equal `n_blocks`
                Allowed styles: "basic". "slice", "flash", "memeff", "slice_memeff".
            dropouts : Tuple[float, ...], default=(False, False)
                Dropout probabilities for the SelfAttention blocks.
            biases : bool, default=(True, True)
                Include bias terms in the SelfAttention blocks.
            layer_scales : bool, default=(False, False)
                Learnable layer weights for the self-attention matrix.
            mlp_ratio : int, default=2
                Scaling factor for the number of input features to get the number of
                hidden features in the final `Mlp` layer of the transformer.
            slice_size : int, optional, default=4
                Slice size for sliced self-attention. This is used only if
                `name = "slice"` for a SelfAttentionBlock.
            **kwargs:
                Arbitrary key-word arguments.

        Raises
        ------
            ValueError: If the lengths of the tuple args are not equal to `n_blocks`.
        """
        super().__init__()

        illegal_args = [
            (k, a)
            for k, a in locals().items()
            if isinstance(a, tuple) and len(a) != n_blocks
        ]

        if illegal_args:
            raise ValueError(
                f"All the tuple-arg lengths need to be equal to `n_blocks`={n_blocks}. "
                f"Illegal args: {illegal_args}"
            )

        self.tr_blocks = nn.ModuleList()
        self.layer_scales = nn.ModuleList()
        blocks = list(range(n_blocks))
        for i in blocks:
            cross_dim = cross_attention_dim if i == blocks[-1] else None

            att_block = SelfAttentionBlock(
                name=block_types[i],
                how=computation_types[i],
                query_dim=query_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                cross_attention_dim=cross_dim,
                dropout=dropouts[i],
                biases=biases[i],
                slice_size=slice_size,
                **kwargs,
            )
            self.tr_blocks.append(att_block)

            # add layer scale. (Optional)
            ls = LayerScale(query_dim) if layer_scales[i] else Identity()
            self.layer_scales.append(ls)

        self.mlp = MlpBlock(
            in_channels=query_dim,
            mlp_ratio=mlp_ratio,
            activation=activation,
            normalization="ln",
            norm_kwargs={"normalized_shape": query_dim},
            act_kwargs=kwargs,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the transformer layer.

        Parameters
        ----------
            x : torch.Tensor
                Input query. Shape (B, H*W, query_dim)).
            context : torch.Tensor, optional
                Context tensor. Shape (B, H*W, query_dim)). Can be used to add context
                information from another source than the current layer.

        Returns
        -------
            torch.Tensor:
                Self-attended input tensor. Shape (B, H*W, query_dim).
        """
        n_blocks = len(self.tr_blocks)

        for i, (tr_block, ls) in enumerate(zip(self.tr_blocks, self.layer_scales), 1):
            # apply context only at the last transformer block
            con = None
            if i == n_blocks:
                con = context

            x = tr_block(x, con)
            x = ls(x)

        return self.mlp(x) + x
