from typing import Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.modules import Conv, Norm

__all__ = ["SumBlock"]


class SumBlock(nn.ModuleDict):
    def __init__(
        self,
        in_channels: int,
        skip_channels: Tuple[int, ...],
        convolution: str = "conv",
        normalization: str = "bn",
        **kwargs,
    ) -> None:
        """Merge by summation.

        Handles clashing channel numbers with a regular conv block.

        Parameters
        ----------
            in_channels : int
                Number of input channels
            skip_channels : Tuple[int, ...]:
                Number of skip channels
            convolution : str
                Name of the convolution method in the downsampling blocks.
            normalization : str
                Name of the normalization method in the downsampling blocks.
        """
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.pool = [chl != self.in_channels for chl in self.skip_channels]

        # add channel pooling modules if necessary
        for i, needs_pooling in enumerate(self.pool):
            if needs_pooling:
                downsample = nn.Sequential(
                    Conv(
                        convolution,
                        in_channels=skip_channels[i],
                        bias=False,
                        out_channels=in_channels,
                        kernel_size=1,
                        padding=0,
                    ),
                    Norm(normalization, num_features=in_channels),
                )
                self.add_module(f"downsample{i + 1}", downsample)

    @property
    def out_channels(self) -> int:
        """Out channels."""
        return self.in_channels

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward of the sum block.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Shape: (B, C, H, W)
            skips : Tuple[torch.Tensor, ...]
                All the skip features in a list. Shapes: (B, C, H, W).

        Returns
        -------
            torch.Tensor:
                The summed output tensor. Shape (B, C, H, W).
        """
        if self.values():
            skips = list(skips)
            for i, needs_pooling in enumerate(self.pool):
                if needs_pooling:
                    skips[i] = self[f"downsample{i + 1}"](skips[i])

        x = torch.stack([x, *skips], dim=0).sum(dim=0)

        return x

    def extra_repr(self) -> str:
        """Add extra info to print."""
        s = "in_channels={in_channels}, skip_channels={skip_channels}"
        s = s.format(**self.__dict__) + f", out_channels={self.out_channels}"
        return s
