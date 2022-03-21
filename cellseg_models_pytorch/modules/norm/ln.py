import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/norm.py
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_features: int, **kwargs) -> None:
        """Layernorm wrap for BCHW shaped tensors.

        Parameters
        ----------
            num_features : int
                Number of input channels/features.
        """
        super().__init__(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of 2D layer norm."""
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 3, 1, 2)
