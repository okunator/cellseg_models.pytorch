"""Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py # noqa
Small mods to docstrings and code to match the style of the rest of the project.

Copyright 2023 Oskari Lehtonen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseTrEncoder

__all__ = ["ImageEncoderViT", "VitDetSAM", "build_sam_encoder"]


# name to pre-trained sam weights mapping
MODEL_URLS = {
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # noqa
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # noqa
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # noqa
}


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """Create the SAM ViT backbone.

        Parameters
        ----------
            img_size : int, default=1024
                Input image size.
            patch_size : int, default=16
                Patch size.
            in_chans : int, default=3
                Number of input image channels.
            embed_dim : int, default=768
                Patch embedding dimension.
            depth : int, default=12
                Depth of ViT.
            num_heads : int, default=12
                Number of attention heads in each ViT block.
            mlp_ratio : float, default=4.0
                Ratio of mlp hidden dim to embedding dim.
            qkv_bias : bool, default=True
                If True, add a learnable bias to query, key, value.
            norm_layer : nn.Module, default=nn.LayerNorm
                Normalization layer.
            act_layer : nn.Module
                Activation layer. Default: nn.GELU.
            use_abs_pos : bool, default=True
                If True, use absolute positional embeddings.
            use_rel_pos : bool, default=False
                If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init : bool, default=True
                If True, zero initialize relative positional parameters.
            window_size : int, default=0
                Window size for window attention blocks.
            global_attn_indexes : Tuple[int, ...], default=()
                Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the backbone."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Create Transformer block w support of window attn & residual propagation.

        Parameters
        ----------
            dim : int
                Number of input channels.
            num_heads : int
                Number of attention heads in each ViT block.
            mlp_ratio : float, default=4.0
                Ratio of mlp hidden dim to embedding dim.
            qkv_bias : bool, default=True
                If True, add a learnable bias to query, key, value.
            norm_layer : nn.Module, default=nn.LayerNorm
                Normalization layer.
            act_layer : nn.Module, default=nn.GELU
                Activation layer.
            use_rel_pos : bool, default=False
                If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init : bool, default=True
                If True, zero initialize relative positional parameters.
            window_size : int, default=0
                Window size for window attention blocks. If it equals 0, then use global
                attention.
            input_size : Tuple[int, int], optional
                Input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        """Create Multi-head Attention block with relative position embeddings.

        Parameters
            dim : int
                Number of input channels.
            num_heads : int, default=8
                Number of attention heads.
            qkv_bias : bool, default=True
                If True, add a learnable bias to query, key, value.
            rel_pos : bool, default=False
                If True, add relative positional embeddings to the attention map.
            input_size : Tuple[int, int], optional
                Input resolution for calculating the relative positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention block."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Parameters
    ----------
        x : torch.Tensor
            Input tokens with [B, H, W, C].
        window_size : int
            Window size.

    Returns
    -------
    Tuple[torch.Tensor, Tuple[int, int]]:
        - windows after partition with [B * num_windows, window_size, window_size, C].
        - padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Window unpartition into original sequences and removing padding.

    Parameters
    ----------
        windows : torch.Tensor
            Input tokens with [B * num_windows, window_size, window_size, C].
        window_size : int
            Window size.
        pad_hw : Tuple[int, int]
            Padded height and width (Hp, Wp).
        hw : Tuple[int, int]
            Original height and width (H, W) before padding.

    Returns
    -------
        torch.Tensor:
            Unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Get relative positional embeddings.

    According to the relative positions of query and key sizes.

    Parameters
    ----------
        q_size : int
            Size of query q.
        k_size : int
            Size of key k.
        rel_pos : torch.Tensor
            Relative position embeddings (L, C).

    Returns
    -------
        torch.Tensor:
            Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950

    Parameters
    ----------
        attn : torch.Tensor
            Attention map.
        q : torch.Tensor
            Query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h : torch.Tensor
            Relative position embeddings (Lh, C) for height axis.
        rel_pos_w : torch.Tensor
            Relative position embeddings (Lw, C) for width axis.
        q_size : Tuple[int, int]
            Spatial sequence size of query q with (q_h, q_w).
        k_size : Tuple[int, int]
            Spatial sequence size of key k with (k_h, k_w).

    Returns
    -------
        torch.Tensor:
            Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Image to Patch Embedding.

        Parameters
        ----------
            kernel_size : Tuple[int, int], default=(16, 16)
                Kernel size of the projection layer.
            stride Tuple[int, int], default=(16, 16)
                Stride of the projection layer.
            padding : Tuple[int, int], default=(0, 0)
                Padding size of the projection layer.
            in_chans : int, default=3
                Number of input image channels.
            embed_dim : int, default=768
                Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the patch embedding."""
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """MLP block.

        Parameters
        ----------
            embedding_dim : int
                Embedding dimension.
            mlp_dim : int
                Hidden dimension of the MLP.
            act : Type[nn.Module], default=nn.GELU
                Activation layer.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP block."""
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        """Layer normalization for 2D data.

        Parameters
        ----------
            num_channels : int
                Number of channels.
            eps : float, default=1e-6
                Epsilon value for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# Simple image-encoder wrapper
class VitDetSAM(BaseTrEncoder):
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        out_indices: Optional[Tuple[int, ...]] = None,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        **kwargs,
    ) -> None:
        """Create a wrapper for the SAM ViT backbone.

        Parameters
        ----------
        checkpoint_path : Optional[Union[str, Path]], optional
            Path to the weights of the backbone. If None and pretrained is False the
            backbone is initialized randomly. Defaults to None.
        out_indices : Optional[Tuple[int, ...]], optional
            Indices of the intermediate features to return. If None, only the last
            features are returned. Defaults to None.
        img_size : int, default=1024
            Input image size.
        patch_size : int, default=16
            Patch size.
        in_chans : int, default=3
            Number of input image channels.
        embed_dim : int, default=768
            Patch embedding dimension.
        depth : int, default=12
            Depth of ViT.
        num_heads : int, default=12
            Number of attention heads in each ViT block.
        mlp_ratio : float, default=4.0
            Ratio of mlp hidden dim to embedding dim.
        out_chans : int, default=256
            Output channels of the neck.
        qkv_bias : bool, default=True
            If True, add a learnable bias to query, key, value.
        norm_layer : nn.Module, default=nn.LayerNorm
            Normalization layer.
        act_layer : nn.Module, default=nn.GELU
            Activation layer.
        use_abs_pos : bool, default=True
            If True, use absolute positional embeddings.
        use_rel_pos : bool, default=False
            If True, add relative positional embeddings to the attention map.
        rel_pos_zero_init : bool, default=True
            If True, zero initialize relative positional parameters.
        window_size : int, default=0
            Window size for window attention blocks.
        global_attn_indexes : Tuple[int, ...], default=()
            Indexes for blocks using global attention.
        """
        super().__init__(
            name="SAM-VitDet",
            checkpoint_path=checkpoint_path,
            out_indices=out_indices,
        )
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.in_channels = in_chans

        self.backbone = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_abs_pos=use_abs_pos,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
        )

        if checkpoint_path is not None:
            self.load_checkpoint()

    def _strip_state_dict(self, state_dict: Dict) -> OrderedDict:
        """Strip the 'image_encoder' from the SAM state dict keys."""
        new_dict = {}
        for k, w in state_dict.items():
            if "image_encoder" in k:
                spl = ["".join(kk) for kk in k.split(".")]
                new_key = ".".join(spl[1:])
                new_dict[new_key] = w

        return new_dict

    @property
    def n_blocks(self):
        """Get the number of attention blocks in the backbone."""
        return len(self.backbone.blocks)

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass of the backbone and return all the features.

        Forward pass of the backbone and return all the features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (input image). Shape: (B, C, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
            torch.Tensor: Output of last layers (all tokens, without classification)
            torch.Tensor: Classification output
            torch.Tensor: All the intermediate features from the attention blocks
        """
        x = self.backbone.patch_embed(x)

        if self.backbone.pos_embed is not None:
            token_size = x.shape[1]
            x = x + self.backbone.pos_embed[:, :token_size, :token_size, :]

        # collect intermediate outputs and add cls token block
        intermediate_outputs = []
        for i, blk in enumerate(self.backbone.blocks, 1):
            x = blk(x)
            if i in self.out_indices:
                intermediate_outputs.append(x.permute(0, 3, 1, 2))

        output = self.backbone.neck(x.permute(0, 3, 1, 2))
        _output = output.reshape(*(*output.shape[:2], -1))

        return torch.mean(_output, axis=-1), output, intermediate_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the backbone."""
        logits, cls_token, features = self.forward_features(x)

        return features


def build_sam_encoder(
    name: str, pretrained: bool = True, checkpoint_path: str = None, **kwargs
) -> nn.Module:
    """Build a SAM ViT encoder.

    Parameters
    ----------
    name : str
        Name of the encoder. Must be one of "sam_vit_h", "sam_vit_l", "sam_vit_b".
    pretrained : bool, optional
        If True, load pretrained weights, by default True.
    checkpoint_path : str, optional
        Path to the weights of the backbone. If None, the backbone is initialized
        randomly. Defaults to None.

    Returns
    -------
    nn.Module
        The initialized SAM ViT encoder.
    """
    if name not in ("sam_vit_h", "sam_vit_l", "sam_vit_b"):
        raise ValueError(
            f"Unknown encoder name: {name}, "
            "allowed values are 'sam_vit_h', 'sam_vit_l', 'sam_vit_b'"
        )

    if checkpoint_path is None and pretrained:
        checkpoint_path = MODEL_URLS[name]

    if name == "sam_vit_h":
        sam_vit = VitDetSAM(
            checkpoint_path=checkpoint_path,
            out_indices=[8, 16, 24, 32],
            global_attn_indexes=[7, 15, 23, 31],
            num_heads=16,
            depth=32,
            embed_dim=1280,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=False,
            window_size=14,
        )
    elif name == "sam_vit_l":
        sam_vit = VitDetSAM(
            checkpoint_path=checkpoint_path,
            out_indices=[6, 12, 18, 24],
            global_attn_indexes=[5, 11, 17, 23],
            num_heads=16,
            depth=24,
            embed_dim=1024,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=False,
            window_size=14,
        )
    elif name == "sam_vit_b":
        sam_vit = VitDetSAM(
            checkpoint_path=checkpoint_path,
            out_indices=[3, 6, 9, 12],
            global_attn_indexes=[2, 5, 8, 11],
            num_heads=12,
            depth=12,
            embed_dim=768,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=False,
            window_size=14,
        )

    return sam_vit
