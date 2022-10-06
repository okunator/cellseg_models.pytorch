import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WSConv2d"]


# Adapted from
# https://github.com/joe-siyuan-qiao/WeightStandardization
class WSConv2d(nn.Conv2d):
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
        eps: float = 1e-7,
        **kwargs
    ) -> None:
        """Weight standardized convolution.

        https://arxiv.org/abs/1903.10520

        Parameters
        ----------
            Refer to nn.Conv2d
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weight standardized convolution forward pass."""
        weight = self.weight

        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )

        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps

        weight = weight / std.expand_as(weight)

        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
