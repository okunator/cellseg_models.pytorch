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
        out_indices: List[int] = None,
        **kwargs
    ) -> None:
        """Import any encoder from timm package.

        Parameters
        ----------
            name : str
                Name of the encoder.
            pretrained : bool, optional
                If True, load pretrained weights, by default True.
            in_channels : int, optional
                Number of input channels, by default 3.
            depth : int, optional
                Number of output features, by default 5.
            out_indices : List[int], optional
                Indices of the output features, by default None. If None, all the
                features are returned.
        """
        super().__init__()

        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        self.model = timm.create_model(name, **kwargs)

        self.out_indices = out_indices
        self.in_channels = in_channels
        self.out_channels = tuple(self.model.feature_info.channels()[::-1])

        if self.out_indices is not None:
            self.out_channels = tuple(self.out_channels[i] for i in self.out_indices)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the encoder and return all the features."""
        features = self.model(x)
        return features[::-1]
