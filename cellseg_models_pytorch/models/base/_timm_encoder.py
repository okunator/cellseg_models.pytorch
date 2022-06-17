from typing import List

import timm
import torch
import torch.nn as nn

__all__ = ["TimmEncoder"]


class TimmEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        **kwargs
    ) -> None:
        """Import any encoder from timm package."""
        super().__init__()

        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        self.model = timm.create_model(name, **kwargs)

        self.in_channels = in_channels
        self.out_channels = tuple(self.model.feature_info.channels()[::-1])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the encoder and return all the features."""
        features = self.model(x)
        return features[::-1]
