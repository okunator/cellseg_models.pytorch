import torch
import torch.nn as nn

__all__ = ["BaseSelfAttention"]


class BaseSelfAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        how: str = "basic",
        slice_size: int = None,
        **kwargs,
    ) -> None:
        """Initialize a base class for self-attention modules.

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
        super().__init__()

        allowed = ("basic", "flash", "slice", "memeff", "slice-memeff")
        if how not in allowed:
            raise ValueError(
                f"Illegal exact self attention type given. Got: {how}. "
                f"Allowed: {allowed}."
            )

        self.how = how
        self.head_dim = head_dim
        self.num_heads = num_heads

        if how == "slice":
            if any(s is None for s in (slice_size, num_heads)):
                raise ValueError(
                    "If `how` is set to 'slice', `slice_size`, `num_heads`, "
                    f"need to be given integer values. Got: `slice_size`: {slice_size} "
                    f"and `num_heads`: {num_heads}."
                )

        if how in ("memeff", "slice-memeff"):
            try:
                import xformers  # noqa F401
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "`self_attention` was set to `memeff`. The method requires the "
                    "xformers package. See how to install xformers: "
                    "https://github.com/facebookresearch/xformers"
                )
            if not torch.cuda.is_available():
                raise ValueError(
                    f"`how` was set to {how}. This method for computing self attentiton"
                    " is implemented  with `xformers.memory_efficient_attention` that "
                    "requires cuda."
                )

        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        self.slice_size = slice_size
        self.proj_channels = self.head_dim * self.num_heads
