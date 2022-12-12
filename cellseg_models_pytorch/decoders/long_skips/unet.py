from typing import Tuple

import torch
import torch.nn as nn

from ...modules import Identity
from .merging import Merge

__all__ = ["UnetSkip"]


class UnetSkip(nn.Module):
    def __init__(
        self,
        stage_ix: int,
        in_channels: int,
        skip_channels: Tuple[int, ...] = None,
        merge_policy: str = "sum",
        **kwargs
    ) -> None:
        """U-net-like skip connection block.

        U-Net: Convolutional Networks for Biomedical Image Segmentation
            - https://arxiv.org/abs/1505.04597#

        Parameters
        ----------
            stage_ix : int
                Index number signalling the current decoder stage
            in_channels : int, default=None
                The number of channels in the input tensor.
            skip_channels : Tuple[int, ...]
                Tuple of the number of channels in the encoder stages.
                Order is bottom up. This list does not include the final
                bottleneck stage out channels. e.g. (1024, 512, 256, 64).
            merge_policy : str, default="sum"
                Sum or concatenate the features together. One of ("sum", "cat").
        """
        super().__init__()
        self.merge_policy = merge_policy
        self.in_channels = in_channels
        self.stage_ix = stage_ix

        self.skip_out_chl = 0
        self.merge = Identity()
        if stage_ix < len(skip_channels):
            self.skip_out_chl = skip_channels[stage_ix]
            self.merge = Merge(
                self.merge_policy,
                in_channels=self.in_channels,
                skip_channels=(self.skip_out_chl,),
                **kwargs
            )

    @property
    def out_channels(self) -> int:
        """Out channels."""
        out_channels = self.in_channels
        if self.merge_policy == "cat":
            out_channels += self.skip_out_chl

        return out_channels

    def forward(
        self, x: torch.Tensor, skips: Tuple[torch.Tensor, ...], **kwargs
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of the U-net long skip connection.

        Parameters
        ----------
            x : torch.Tensor
                Input from the previous decoder layer. Shape (B, C, H, W).
            skips : Tuple[torch.Tensor, ...]
                All the encoder feature maps. Shapes: (B, C, H, W).
            ix : int
                Index of the decoder stage.

        Returns
        -------
            Tuple[torch.Tensor, ...]:
                The skip connection tensor. Shape (B, C ( + n_skip_cahnnels), H, W).
        """
        if self.stage_ix < len(skips):
            skip = skips[self.stage_ix]
            x = self.merge(x, (skip,))

        return x
