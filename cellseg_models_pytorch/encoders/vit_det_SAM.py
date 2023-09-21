from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from ._base import BaseTrEncoder
from .sam import ImageEncoderViT

__all__ = ["VitDetSAM", "build_sam_encoder"]


# name to pre-trained sam weights mapping
MODEL_URLS = {
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # noqa
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # noqa
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # noqa
}


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

        return torch.mean(_output, axis=-1), output, tuple(intermediate_outputs)

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
