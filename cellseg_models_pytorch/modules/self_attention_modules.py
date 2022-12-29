from typing import Tuple

import torch
import torch.nn as nn

from .base_modules import MultiHeadSelfAttention

__all__ = ["SelfAttention", "SelfAttentionBlock"]


class SelfAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        name: str = "exact",
        how: str = "basic",
        cross_attention_dim: int = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        slice_size: int = 4,
        **kwargs,
    ) -> None:
        """Compute self-attention.

        Includes all the data wrangling on the input before attention computation.

        Input Shape: (B, H'*W', query_dim).
        Output Shape: (B, H'*W', query_dim).

        Parameters
        ----------
            query_dim : int
                The number of channels in the query. Typically: num_heads*head_dim
            name : str
                Name of the attention method. One of ("exact", "linformer").
            how : str, default="basic"
                How to compute the self-attention matrix.
                One of ("basic", "flash", "slice", "memeff", "slice-memeff").
                "basic": the normal O(N^2) self attention.
                "flash": the flash attention (by xformers library),
                "slice": batch sliced attention operation to save mem.
                "memeff": xformers.memory_efficient_attention.
                "slice-memeff": Conmbine slicing and memory_efficient_attention.
            cross_attention_dim : int, optional
                Number of channels in the context tensor. Cross attention combines
                asymmetrically two separate embeddings (context and input embeddings).
                E.g. passage from transformer encoder to transformer decoder. If this is
                set to None, no cross attention is applied.
            num_heads : int, default=8
                Number of heads for multi-head attention.
            head_dim : int, default=64
                Number of output channels per head.
            dropout : float, default=0.0
                Dropout probability.
            bias : bool, default=False
                Flag to set bias for Q, K and V.
            slice_size : int, default=4
                Slice size for sliced self-attention. This is used only if
                `self_attention = "slice"`.
            **kwargs:
                Extra key-word arguments for the MHSA-module
        """
        super().__init__()
        self.out_channels = query_dim
        self.num_heads = num_heads
        proj_dim = head_dim * num_heads

        # cross attention dim
        if cross_attention_dim is None:
            cross_attention_dim = query_dim

        self.to_q = nn.Linear(query_dim, proj_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, proj_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, proj_dim, bias=bias)

        self.self_attn = MultiHeadSelfAttention(
            name=name,
            head_dim=head_dim,
            num_heads=self.num_heads,
            how=how,
            slice_size=slice_size,
            **kwargs,
        )

        self.to_out = nn.Linear(proj_dim, query_dim)
        self.dropout = nn.Dropout(dropout) if bool(dropout) else None

    def to_qkv(
        self, features: torch.Tensor, context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get query, key and value tensors.

        NOTE: If a context tensor is given, the key and value will be computed
        from that. If `context=None`, key and value is computed from the input
        features. This allows cross-attention.

        Parameters
        ----------
            features : torch.Tensor
                Input tensor. Usually a projection of the input features into the shape
                of a query tensor. Shape: (B, H*W, proj_dim).
            context : torch.Tensor, optional
                A context tensor. Same shape as `features`.

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The query, key & value tensors.
                Shaped: (B*num_heads, H*W, proj_dim//num_heads).
        """
        query = self.to_q(features)
        context = context if context is not None else features
        key = self.to_k(context)
        value = self.to_v(context)

        query = self._heads2batch(query)
        key = self._heads2batch(key)
        value = self._heads2batch(value)

        return query, key, value

    def _heads2batch(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape `x` input for self-attention.

        NOTE:
        Step 1. Divide the `proj_dim` (num_heads*head_out_channels) long projection
        vectors of pixel vals into `num_heads` (default=8) number of vectors that are
        `ìnner_dim`//`num_heads` (default=64) long.

        Step 2. Then bake the heads dim into batch dimension.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Either, the query, key or value tensor.
                Shape: (B, H*W, proj_dim).

        Returns
        -------
            torch.Tensor
                A reshaped input of shape: (B*num_heads, H*W, proj_dim//num_heads).
        """
        B, seq_len, proj_dim = x.shape
        x = x.reshape(B, seq_len, self.num_heads, proj_dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(
            B * self.num_heads, seq_len, proj_dim // self.num_heads
        )

        return x

    def _batch2heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape the attention scores back to (B, H*W, proj_dim).

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. The attention scores from the self attention computation.
                Shape: (B*num_heads, H*W, proj_dim//num_heads).

        Returns
        -------
            torch.Tensor:
                A reshaped attention score tensor of shape: (B, H*W, proj_dim).
        """
        # num_heads = self.num_heads
        B, seq_len, head_out_channels = x.shape
        x = x.reshape(B // self.num_heads, self.num_heads, seq_len, head_out_channels)
        x = x.permute(0, 2, 1, 3).reshape(
            B // self.num_heads, seq_len, head_out_channels * self.num_heads
        )

        return x

    def forward(
        self, features: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass of the self-attention."""
        # (B, H*W, proj_dim) -> (B*num_heads, H*W, proj_dim//num_heads).
        query, key, value = self.to_qkv(features, context)

        # compute self-attention. Output has same shape as q, k, v
        features = self.self_attn(query, key, value)

        # (B*num_heads, H*W, proj_dim//num_heads) -> (B, H*W, proj_dim)
        features = self._batch2heads(features)

        # linear projection + dropout
        out = self.to_out(features)  # (B, H*W, query_dim)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        how: str,
        query_dim: int,
        name: str = "exact",
        cross_attention_dim: int = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        slice_size: int = 4,
        **kwargs,
    ) -> None:
        """Singular self-attention block.

        These can be stacked to form a tranformer block.

        NOTE: Can be used as a cross-attention block if `cross_attention_dim` is given.

        Input Shape: (B, H'*W', query_dim).
        Output Shape: (B, H'*W', query_dim).

        Parameters
        ----------
            name : str
                Name of the attention method. One of ("exact", "linformer").
            how : str, default="basic"
                How to compute the self-attention matrix.
                One of ("basic", "flash", "slice", "memeff", "slice-memeff").
                "basic": the normal O(N^2) self attention.
                "flash": the flash attention (by xformers library),
                "slice": batch sliced attention operation to save mem.
                "memeff": xformers.memory_efficient_attention.
                "slice-memeff": Conmbine slicing and memory_efficient_attention.
            query_dim : int
                The number of channels in the query. Typically: num_heads*head_dim
            cross_attention_dim : int, optional
                Number of channels in the context tensor. Cross attention combines
                asymmetrically two separate embeddings (context and input embeddings).
                E.g. passage from transformer encoder to transformer decoder. If this is
                set to None, no cross attention is applied.
            num_heads : int, default=8
                Number of heads for multi-head attention.
            head_dim : int, default=64
                Number of output channels per head.
            dropout : float, default=0.0
                Dropout probability.
            bias : bool, default=False
                Flag to set bias for Q, K and V.
            slice_size : int, default=4
                Slice size for sliced self-attention. This is used only if
                `self_attention = "slice"`.
        """
        super().__init__()

        self.norm = nn.LayerNorm(query_dim)
        self.att = SelfAttention(
            name=name,
            how=how,
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            slice_size=slice_size,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of one transformer block."""
        residual = x
        x = self.norm(x)
        x = self.att(x, context) + residual  # residual post-layernorm

        return x
