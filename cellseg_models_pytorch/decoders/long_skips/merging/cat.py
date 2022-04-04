from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["CatBlock"]


class CatBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: Tuple[int, ...],
        **kwargs,
    ) -> None:
        """Merge skip blocks by concatenation.

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            skip_channels : Tuple[int, ...]
                Number of skip channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels

    @property
    def out_channels(self) -> int:
        """Out channels."""
        return sum([self.in_channels] + list(self.skip_channels))

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward of the cat block.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Shape: (B, C, H, W)
            skips : Tuple[torch.Tensor, ...]
                All the skip features in a list. Shapes: (B, C, H, W).

        Returns
        -------
            torch.Tensor:
                The concatenated output tensor. Shape (B, C+skip_channels, H, W).
        """
        return torch.cat([x, *skips], dim=1)

    def extra_repr(self) -> str:
        """Add extra info to print."""
        s = "in_channels={in_channels}, skip_channels={skip_channels}"
        s = s.format(**self.__dict__) + f", out_channels={self.out_channels}"
        return s
