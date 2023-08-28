"""Adapted from https://github.com/jopo666/HistoEncoder.

Copyright 2023 Joona Pohjonen

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

from pathlib import Path
from typing import List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn

from cellseg_models_pytorch.encoders._base import BaseTrEncoder

__all__ = ["HistoEncoder", "build_histo_encoder"]

# histo_encoder model name to timm model name mapping
NAME_TO_MODEL = {
    "histo_encoder_prostate_s": "xcit_small_12_p16_224",
    "histo_encoder_prostate_m": "xcit_medium_24_p16_224",
}

# name to pre-trained weights mapping
MODEL_URLS = {
    "histo_encoder_prostate_s": "https://dl.dropboxusercontent.com/s/tbff9wslc8p7ie3/prostate_small.pth?dl=0",  # noqa
    "histo_encoder_prostate_m": "https://dl.dropboxusercontent.com/s/k1fr09x5auki8sp/prostate_medium.pth?dl=0",  # noqa
}


class HistoEncoder(BaseTrEncoder):
    def __init__(
        self,
        backbone: nn.Module,
        checkpoint_path: Optional[Union[Path, str]] = None,
        out_indices: Optional[Tuple[int, ...]] = None,
        num_blocks: int = 1,
        embed_dim: int = 384,
        patch_size: int = 16,
        avg_pool: bool = False,
        **kwargs,
    ) -> None:
        """Create HistoEncoder backbone.

        HistoEncoder: https://github.com/jopo666/HistoEncoder

        Parameters
        ----------
        checkpoint_path : Optional[Union[Path, str]], optional
            Path to the weights of the backbone. If None and pretrained is False the
            backbone is initialized randomly. Defaults to None.
        num_blocks : int, optional
            Number of attention blocks to include in the extracted features.
            When `num_blocks>1`, the outputs of the last `num_blocks` attention
            blocks are concatenated to make up the features. Defaults to 1.
        avg_pool : bool, optional
            Whether to average pool the outputs of the last attention block.
            Defaults to False.
        """
        super().__init__(
            name="Histo-encoder",
            checkpoint_path=checkpoint_path,
            out_indices=out_indices,
        )

        self.backbone = backbone
        self.avg_pool = avg_pool
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        if checkpoint_path is not None:
            self.load_checkpoint()

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
        B = x.shape[0]
        x, (Hp, Wp) = self.backbone.patch_embed(x)

        if self.backbone.pos_embed is not None:
            pos_encoding = (
                self.backbone.pos_embed(B, Hp, Wp)
                .reshape(B, -1, x.shape[1])
                .permute(0, 2, 1)
            )
            x = x + pos_encoding

        x = self.backbone.pos_drop(x)

        # Collect intermediate outputs.
        intermediate_outputs = []
        res_outputs = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, Hp, Wp)
            intermediate_outputs.append(x)
            if i in self.out_indices:
                res_outputs.append(
                    x.reshape(B, Hp, Wp, self.embed_dim).permute(0, 3, 1, 2)
                )

        # collect intermediate outputs and add cls token block
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for j, blk in enumerate(self.backbone.cls_attn_blocks, i + 1):
            x = blk(x)
            intermediate_outputs.append(x)
            if j in self.out_indices:
                res_outputs.append(
                    x[:, 1:, :].reshape(B, Wp, Hp, self.embed_dim).permute(0, 3, 1, 2)
                )

        norm_outputs = [
            self.backbone.norm(x) for x in intermediate_outputs[-self.num_blocks :]
        ]
        output = torch.cat([x[:, 0] for x in norm_outputs], axis=-1)

        if self.avg_pool:
            output = torch.cat(
                [output, torch.mean(norm_outputs[-1][:, 1:], dim=1)], axis=-1
            )

        return torch.mean(norm_outputs[-1][:, 1:], dim=1), output, res_outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the histo-encoder backbone."""
        logits, cls_token, features = self.forward_features(x)

        return features


def build_histo_encoder(
    name: str, pretrained: bool = True, checkpoint_path: str = None, **kwargs
) -> HistoEncoder:
    """Build HistoEncoder backbone.

    Parameters
    ----------
    name : str
        Name of the encoder. Must be one of "histo_encoder_prostate_s".
        "histo_encoder_prostate_m".
    pretrained : bool, optional
        If True, load pretrained weights, by default True.
    checkpoint_path : str, optional
        Path to the weights of the backbone. If None and pretrained is False the
        backbone is initialized randomly. Defaults to None.

    Returns
    -------
    nn.Module
        The initialized Histo-encoder.
    """
    if name not in ("histo_encoder_prostate_s", "histo_encoder_prostate_m"):
        raise ValueError(
            f"Unknown encoder name: {name}, "
            "allowed values are 'histo_encoder_prostate_s', 'histo_encoder_prostate_m'"
        )

    if checkpoint_path is None and pretrained:
        checkpoint_path = MODEL_URLS[name]

    # init XCit backbone
    backbone = timm.create_model(NAME_TO_MODEL[name], num_classes=0, **kwargs)

    if name == "histo_encoder_prostate_s":
        histo_encoder = HistoEncoder(
            backbone=backbone,
            out_indices=(2, 5, 10, 13),
            checkpoint_path=checkpoint_path,
            embed_dim=384,
            patch_size=16,
        )
    elif name == "histo_encoder_prostate_m":
        histo_encoder = HistoEncoder(
            backbone=backbone,
            out_indices=(4, 11, 20, 25),
            checkpoint_path=checkpoint_path,
            embed_dim=512,
            patch_size=16,
        )

    return histo_encoder
