import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ScaledWSConv2d"]


# Adapted from:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/std_conv.py
class ScaledWSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        gamma: float = 1.0,
        gain_init: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        """Conv2d layer with Scaled Weight Standardization.

        https://arxiv.org/abs/2101.08692

        Parameters
        ----------
            Refer to nn.Conv2d.

            gamma : float
                Fixed constant to preserve the variance of residual-blocks.
            gain_init : float
                Init value for the gain tensor.

        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of scaled ws conv."""
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)

        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
