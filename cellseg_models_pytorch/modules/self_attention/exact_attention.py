import torch
import torch.nn as nn

try:
    from xformers.ops import memory_efficient_attention
except ModuleNotFoundError:
    pass


class ExactSelfAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        self_attention: str = "basic",
        num_heads: int = None,
        slice_size: int = None,
    ) -> None:
        """Compute exact attention.

        Three variants:
        - basic self-attention implementation with torch.matmul O(N^2)
        - slice-attention - Computes the attention matrix in slices to save mem.
        - `xformers.ops.memory_efficient_attention` from xformers package.

        Parameters
        ----------
            head_dim : int
                Out dim per attention head.
            self_attention : str, default="basic"
                One of ("basic", "flash", "sliced", "memeff").
                "basic": the normal O(N^2) self attention.
                "flash": the flash attention (by xformers library),
                "slice": batch sliced attention operation to save mem.
                "memeff" xformers.memory_efficient_attention.
            num_heads : int, optional
                Number of heads. Used only if `slice_attention = True`.
            slice_size, int, optional
                The size of the slice. Used only if `slice_attention = True`.

        Raises
        ------
            - ValueError:
                - If illegal self attention method is given.
                - If `self_attention` is set to `slice` while `num_heads` | `slice_size`
                    args are not given proper integer values.
                - If `self_attention` is set to `memeff` but cuda is not available.
            - ModuleNotFoundError:
                - If `self_attention` is set to `memeff` and `xformers` package is not
                installed
        """
        super().__init__()

        allowed = ("basic", "flash", "slice", "memeff")
        if self_attention not in allowed:
            raise ValueError(
                f"Illegal exact self attention type given. Got: {self_attention}. "
                f"Allowed: {allowed}."
            )

        self.self_attention = self_attention
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = head_dim**-0.5

        # These are only used for slice attention.
        if self_attention == "slice":
            # for slice_size > 0 the attention score computation
            # is split across the batch axis to save memory
            self.slice_size = slice_size
            self.proj_channels = self.head_dim * self.num_heads
            if any(s is None for s in (slice_size, num_heads)):
                raise ValueError(
                    "If `slice_attention` is set to True, `slice_size`, `num_heads`, "
                    f"need to be given integer values. Got: `slice_size`: {slice_size} "
                    f"and `num_heads`: {num_heads}."
                )

        if self_attention == "memeff":
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
                    "`self_attention` was set to `memeff`. The method is implemented "
                    "with `xformers.memory_efficient_attention` that requires cuda."
                )

    def _attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute exact self attention with torch. Complexity: O(N**2).

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
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        probs = scores.softmax(dim=-1)

        # compute attention output
        return torch.matmul(probs, value)

    def _slice_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Compute exact attention in slices to save memory.

        NOTE: adapted from hugginface diffusers package. Their implementation
        just dont handle the case where B // slize_size doesn't divide evenly.
        This would end up in zero-matrices at the final batch dimensions of
        the batched attention matrix.

        NOTE: The input is sliced in the batch dimension.

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
        B, seq_len = query.shape[:2]
        out = torch.zeros(
            (B, seq_len, self.proj_channels // self.num_heads),
            device=query.device,
            dtype=query.dtype,
        )

        # get the modulo if B/slice_size is not evenly divisible.
        n_slices, mod = divmod(out.shape[0], self.slice_size)
        if mod != 0:
            n_slices += 1

        it = list(range(n_slices))
        for i in it:
            start = i * self.slice_size
            end = (i + 1) * self.slice_size

            if i == it[-1]:
                end = start + mod

            attn_slice = torch.matmul(query[start:end], key[start:end].transpose(1, 2))
            attn_slice *= self.scale

            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start:end])

            out[start:end] = attn_slice

        return out

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
        if self.self_attention == "memeff":
            if all([query.is_cuda, key.is_cuda, value.is_cuda]):
                attn = memory_efficient_attention(query, key, value)
            else:
                raise RuntimeError(
                    "`xformers.ops.memory_efficient_attention` is only implemented "
                    "for cuda. Make sure your inputs & model devices are set to cuda."
                )
        elif self.self_attention == "flash":
            raise NotImplementedError
        elif self.self_attention == "slice":
            attn = self._slice_attention(query, key, value)
        else:
            attn = self._attention(query, key, value)

        return attn

    def __repr__(self) -> str:
        """Add extra info to print."""
        s = "ExactSelfAttention(self_attention='{self_attention}', head_dim={head_dim}, num_heads={num_heads})"  # noqa: E501
        s = s.format(**self.__dict__)
        return s
