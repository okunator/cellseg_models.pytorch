import torch

try:
    from xformers.ops import memory_efficient_attention

    _has_xformers = True
except ModuleNotFoundError:
    _has_xformers = False


__all__ = ["multihead_attention", "slice_mha", "mha", "compute_mha"]


def multihead_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
    **kwargs,
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
        scale : float, optional
            Scaling factor for Q @ K'. If None, query.shape[-1]**-0.5 will
            be used

    Returns
    -------
        torch.Tensor:
            The self-attention matrix. Same shape as inputs.
    """
    if scale is None:
        scale = query.shape[-1] ** -0.5

    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    probs = scores.softmax(dim=-1)

    # compute attention output
    return torch.matmul(probs, value)


def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    att_type: str = "basic",
    **kwargs,
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
        att_type : str, default="basic"
            The type of the self-attention computation.
            One of: ("basic", "flash", "memeff").
        **kwargs:
            Extra key-word arguments for the mha computation.

    Returns
    -------
        torch.Tensor:
            The self-attention matrix. Same shape as inputs.
    """
    if att_type == "memeff":
        if _has_xformers:
            if all([query.is_cuda, key.is_cuda, value.is_cuda]):
                attn = memory_efficient_attention(query, key, value)
            else:
                raise RuntimeError(
                    "`xformers.ops.memory_efficient_attention` is only implemented "
                    "for cuda. Make sure your inputs & model devices are set to cuda."
                )
        else:
            raise ModuleNotFoundError(
                "Trying to use `memory_efficient_attention`. The method requires the "
                "xformers package. See how to install xformers: "
                "https://github.com/facebookresearch/xformers"
            )
    elif att_type == "flash":
        raise NotImplementedError
    elif att_type == "basic":
        attn = multihead_attention(query, key, value, **kwargs)
    else:
        raise ValueError(
            f"Unknown `att_type` given. Got: {att_type}. "
            f"Allowed: {('memeff', 'flash', 'basic')}"
        )

    return attn


def slice_mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    proj_channels: int,
    num_heads: int,
    slice_size: int = 4,
    att_type: str = "basic",
    **kwargs,
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
        proj_channels : int
            Number of out channels in the token projections.
        num_heads : int
            Number of heads in the mha.
        slice_size : int, default=4
            The size of the batch dim slice.
        att_type : str, default="basic"
            The type of the self-attention computation.
            One of: ("memeff", "slice-memeff").

    Returns
    -------
        torch.Tensor:
            The self-attention matrix. Same shape as inputs.
    """
    allowed = ("slice", "slice-memeff")
    if att_type not in allowed:
        raise ValueError(
            f"Illegal slice-attention given. Got: {att_type}. Allowed: {allowed}."
        )

    # parse the attention type arg
    a = att_type.split("-")
    if len(a) == 1:
        att_type = "basic"
    else:
        att_type = "memeff"

    B, seq_len = query.shape[:2]
    out = torch.zeros(
        (B, seq_len, proj_channels // num_heads),
        device=query.device,
        dtype=query.dtype,
    )

    # get the modulo if B/slice_size is not evenly divisible.
    n_slices, mod = divmod(out.shape[0], slice_size)
    if mod != 0:
        n_slices += 1

    it = list(range(n_slices))
    for i in it:
        start = i * slice_size
        end = (i + 1) * slice_size

        if i == it[-1]:
            end = start + mod

        attn_slice = mha(
            query[start:end], key[start:end], value[start:end], att_type=att_type
        )

        out[start:end] = attn_slice
        del attn_slice
        torch.cuda.empty_cache()

    return out


def compute_mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    how: str,
    **kwargs,
) -> torch.Tensor:
    """Wrap all the different attention matrix computation types under this.

    Parameters
    ----------
        query : torch.Tensor
            Query tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).
        key : torch.Tensor
            Key tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).
        value : torch.Tensor
            Value tensor. Shape: (B*num_heads, H*W, proj_dim//num_heads).
        how : str, default="basic"
            How to compute the self-attention matrix.
            One of ("basic", "flash", "slice", "memeff", "slice-memeff").
            "basic": the normal O(N^2) self attention.
            "flash": the flash attention (by xformers library),
            "slice": batch sliced attention operation to save mem.
            "memeff": xformers.memory_efficient_attention.
            "slice-memeff": Conmbine slicing and memory_efficient_attention.
        **kwargs:
            Extra key-word args for the attention matrix computation.
    """
    allowed = ("basic", "flash", "slice", "memeff", "slice-memeff")
    if how not in allowed:
        raise ValueError(
            f"Illegal exact self attention type given. Got: {how}. "
            f"Allowed: {allowed}."
        )

    if how == "basic":
        attn = mha(query, key, value, att_type="basic", **kwargs)
    elif how == "memeff":
        attn = mha(query, key, value, att_type="memeff", **kwargs)
    elif how == "slice":
        attn = slice_mha(query, key, value, att_type="slice", **kwargs)
    elif how == "slice-memeff":
        attn = slice_mha(query, key, value, att_type="slice-memeff", **kwargs)
    elif how == "flash":
        raise NotImplementedError
    elif how == "slice-flash":
        raise NotImplementedError

    return attn
