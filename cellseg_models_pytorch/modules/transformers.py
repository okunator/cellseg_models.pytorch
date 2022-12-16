from typing import Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.modules import SelfAttentionBlock

from .base_modules import TransformerAct
from .misc_modules import Proj2Attention

__all__ = ["Transformer2D", "TransformerLayer"]


class Transformer2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
        cross_attention_dim: int = None,
        n_blocks: int = 2,
        block_types: Tuple[str, ...] = ("basic", "basic"),
        dropouts: Tuple[float, ...] = (0.0, 0.0),
        biases: Tuple[bool, ...] = (False, False),
        act: str = "geglu",
        num_groups: int = 32,
        slice_size: int = 4,
        fc_projection_mult: int = 4,
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
            block_types : Tuple[str, ...], default=("basic", "basic")
                The name of the SelfAttentionBlocks in the TransformerLayer.
                Length of the tuple has to equal `n_blocks`
                Allowed names: "basic". "slice", "flash".
            dropouts : Tuple[float, ...], default=(False, False)
                Dropout probabilities for the SelfAttention blocks.
            biases : bool, default=(True, True)
                Include bias terms in the SelfAttention blocks.
            act : str, default="geglu"
                The activation function applied at the end of the transformer layer fc.
                One of ("geglu", "approximate_gelu").
            num_groups : int, default=32
                Number of groups in the first group-norm op before the input is
                projected to be suitable for self-attention.
            slice_size : int, default=4
                Slice size for sliced self-attention. This is used only if
                `name = "slice"` for a SelfAttentionBlock.
            fc_projection_mult : int, default=4
                Multiplier that defines the out dimension of the final fc projection
                layer.
        """
        super().__init__()
        self.proj_in = Proj2Attention(
            in_channels=in_channels,
            num_groups=num_groups,
            head_dim=head_dim,
            num_heads=num_heads,
        )

        self.transformer = TransformerLayer(
            query_dim=self.proj_in.proj_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
            n_blocks=n_blocks,
            block_types=block_types,
            dropouts=dropouts,
            biases=biases,
            act=act,
            slice_size=slice_size,
            fc_projection_mult=fc_projection_mult,
        )

        self.proj_out = nn.Conv2d(
            self.proj_in.proj_dim, in_channels, kernel_size=1, stride=1, padding=0
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

        # 1. project
        x = self.proj_in(x)

        # 2. transformer
        x = self.transformer(x, context)

        # 3. Reshape back to image-like shape and project to original input channels.
        x = x.reshape(B, H, W, self.proj_in.proj_dim).permute(0, 3, 1, 2)
        x = self.proj_out(x)

        # 4. residual
        return x + residual


class TransformerLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        cross_attention_dim: int = None,
        act: str = "geglu",
        n_blocks: int = 2,
        block_types: Tuple[str, ...] = ("basic", "basic"),
        dropouts: Tuple[float, ...] = (0.0, 0.0),
        biases: Tuple[bool, ...] = (False, False),
        slice_size: int = 4,
        fc_projection_mult: int = 4,
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
            act : str, default="geglu"
                The activation function applied at the end of the transformer layer fc.
                One of ("geglu", "approximate_gelu").
            n_blocks : int, default=2
                Number of SelfAttentionBlocks used in this layer.
            block_types : Tuple[str, ...], default=("basic", "basic")
                The name of the SelfAttentionBlocks in the TransformerLayer.
                Length of the tuple has to equal `n_blocks`
                Allowed names: "basic". "slice", "flash".
            dropouts : Tuple[float, ...], default=(False, False)
                Dropout probabilities for the SelfAttention blocks.
            biases : bool, default=(True, True)
                Include bias terms in the SelfAttention blocks.
            slice_size : int, default=4
                Slice size for sliced self-attention. This is used only if
                `name = "slice"` for a SelfAttentionBlock.
            fc_projection_mult : int, default=4
                Multiplier that defines the out dimension of the final fc projection
                layer.

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

        self.tr_blocks = nn.ModuleDict()
        blocks = list(range(n_blocks))
        for i in blocks:
            cross_dim = cross_attention_dim if i == blocks[-1] else None

            att_block = SelfAttentionBlock(
                name=block_types[i],
                query_dim=query_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                cross_attention_dim=cross_dim,
                dropout=dropouts[i],
                biases=biases[i],
                slice_size=slice_size,
            )
            self.tr_blocks[f"transformer_{block_types[i]}_{i + 1}"] = att_block

        proj_dim = int(query_dim * fc_projection_mult)
        self.fc = nn.Sequential(
            nn.LayerNorm(query_dim),
            TransformerAct(act, dim_in=query_dim, dim_out=proj_dim),
            nn.Linear(proj_dim, query_dim),
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
        for i, tr_block in enumerate(self.tr_blocks.values(), 1):
            # apply context only at the last transformer block
            con = None
            if i == n_blocks:
                con = context

            x = tr_block(x, con)

        return self.fc(x) + x
