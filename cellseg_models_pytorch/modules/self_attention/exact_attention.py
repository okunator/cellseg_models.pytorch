import torch

from .attention_ops import compute_mha
from .base_attention import BaseSelfAttention

__all__ = ["ExactSelfAttention"]


class ExactSelfAttention(BaseSelfAttention):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        how: str = "basic",
        slice_size: int = None,
        **kwargs,
    ) -> None:
        """Compute exact attention.

        Four variants:
        - basic: self-attention implementation with torch.matmul O(N^2)
        - slice-attention: Computes the attention matrix in slices to save mem.
        - memeff: `xformers.ops.memory_efficient_attention` from xformers package.
        - slice-memeff-attention: Comnbines slice-attention and memeff

        Parameters
        ----------
            head_dim : int
                Out dim per attention head.
            num_heads : int
                Number of heads.
            how : str, default="basic"
                How to compute the self-attention matrix.
                One of ("basic", "flash", "slice", "memeff", "slice-memeff").
                "basic": the normal O(N^2) self attention.
                "flash": the flash attention (by xformers library),
                "slice": batch sliced attention operation to save mem.
                "memeff": xformers.memory_efficient_attention.
                "slice-memeff": Conmbine slicing and memory_efficient_attention.
            slice_size, int, optional
                The size of the slice. Used only if `how in ('slice', 'slice_memeff)`.

        Raises
        ------
            - ValueError:
                - If illegal self attention (`how`) method is given.
                - If `how` is set to `slice` while `num_heads` | `slice_size`
                    args are not given proper integer values.
                - If `how` is set to `memeff` or `slice_memeff` but cuda is not
                    available.
            - ModuleNotFoundError:
                - If `self_attention` is set to `memeff` and `xformers` package is not
                installed
        """
        super().__init__(
            head_dim=head_dim,
            num_heads=num_heads,
            how=how,
            slice_size=slice_size,
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute exact self-attention.

        I.e softmax(Q @ K'/sqrt(head_dim)) @ V

        Parameters
        ----------
            query : torch.Tensor
                Query tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).
            key : torch.Tensor
                Key tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).
            value : torch.Tensor
                Value tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).

        Returns
        -------
            torch.Tensor:
                The self-attention matrix. Same shape as inputs.
        """
        attn = compute_mha(
            query,
            key,
            value,
            self.how,
            slice_size=self.slice_size,  # used only for slice-att
            num_heads=self.num_heads,  # used only for slice-att
            proj_channels=self.proj_channels,  # used only for slice-att
        )

        return attn

    def __repr__(self) -> str:
        """Add extra info to print."""
        s = "ExactSelfAttention(how='{how}', head_dim={head_dim}, num_heads={num_heads})"  # noqa: E501
        s = s.format(**self.__dict__)
        return s
