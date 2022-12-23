import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_ops import compute_mha
from .base_attention import BaseSelfAttention

__all__ = ["LinformerAttention"]


class LinformerAttention(BaseSelfAttention):
    def __init__(
        self,
        seq_len: int,
        head_dim: int,
        num_heads: int,
        k: int = None,
        how: str = "basic",
        slice_size: int = None,
        **kwargs
    ) -> None:
        """Linformer attention mechanism.

        Linformer: Self-Attention with Linear Complexity
        - https://arxiv.org/abs/2006.04768v2

        Adapted from xformers library

        NOTE: Weirdly, even when computing linformer attention with xformers
        `memory_efficient_attention`, linformer needs more memory for long sequences
        (due to the linear layers) than computing exact `memory_efficient_attention`.

        Parameters
        ----------
            seq_len : int
                The length of the sequence. (For per-pixel patches H*W).
            head_dim : int
                Out dim per attention head.
            num_heads : int
                Number of heads.
            k : int, optional
                Divisor for key and value matrices to get low-rank attention matrix.
            how : str, default="basic"
                How to compute the self-attention matrix.
                One of ("basic", "flash", "slice", "memeff", "slice_memeff").
                "basic": the normal O(N^2) self attention.
                "flash": the flash attention (by xformers library),
                "slice": batch sliced attention operation to save mem.
                "memeff": xformers.memory_efficient_attention.
                "slice_memeff": Conmbine slicing and memory_efficient_attention.
            slice_size, int, optional
                The size of the slice. Used only if `how in ('slice', 'slice_memeff)`.
        """
        super().__init__(
            head_dim=head_dim,
            num_heads=num_heads,
            how=how,
            slice_size=slice_size,
        )

        if k is None:
            k = seq_len // 4

        self.k = k
        self.E = nn.Linear(seq_len, k, bias=False)
        self.F = nn.Linear(seq_len, k, bias=False)
        self.seq_len = seq_len

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward pass of the linformer attention mechanism."""
        padding = 0
        if query.shape[1] < self.seq_len:
            padding = self.seq_len - query.shape[1]
            pad_dims = (0, 0, 0, padding)
            query = F.pad(query, pad_dims)
            key = F.pad(key, pad_dims)
            value = F.pad(value, pad_dims)

        key_proj = self.E(key.transpose(-1, -2)).transpose(-1, -2).contiguous()
        value_proj = self.F(value.transpose(-1, -2)).transpose(-1, -2).contiguous()

        out = compute_mha(
            query,
            key_proj,
            value_proj,
            self.how,
            slice_size=self.slice_size,  # used only for slice-att
            num_heads=self.num_heads,  # used only for slice-att
            proj_channels=self.proj_channels,  # used only for slice-att
        )

        return out[:, :-padding, :] if padding > 0 else out
