import torch
import torch.nn as nn

from .base_modules import Conv, Norm

__all__ = ["ChannelPool", "StyleBlock", "StyleReshape", "Proj2Attention"]


class Proj2Attention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_groups: int = 32,
        head_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        """Project image-like data (B, C, H, W) to right shape for a transformer.

        Output shape: (B, H*W, head_dim*num_heads)

        NOTE: Group normalizes input before projecting and reshaping.

        Parameters
        ----------
            in_channels : int
                Number of input channels in the input tensor.
            num_groups : int, default=32
                Number of groups for the group normalization.
            head_dim : int, default=64
                Number of channels per each head.
            num_heads : int, default=8
                Number of heads in multi-head self-attention.
        """
        super().__init__()
        self.proj_dim = head_dim * num_heads
        self.norm = Norm("gn", num_features=in_channels, num_groups=num_groups)
        self.proj_in = nn.Conv2d(
            in_channels, self.proj_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for projection."""
        B, _, H, W = x.shape

        x = self.norm(x)
        projection = self.proj_in(x)

        # reshape to a query. Every pixel value has been projected into a `proj_dim`
        # long vector. I.e every pixel value is represented by a projection vector.
        projection = projection.permute(0, 2, 3, 1).reshape(
            B, H * W, self.proj_dim
        )  # (B, H*W, proj_dim)

        return projection


class ChannelPool(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: str = "bn",
        convolution: str = "conv",
        **kwargs,
    ) -> None:
        """Channel pooling/downsampling module for convenience.

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            convolution : str, default="conv"
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj = nn.Sequential(
            Conv(
                convolution,
                in_channels=in_channels,
                bias=False,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
            Norm(normalization, num_features=self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the channel pool."""
        return self.proj(x)


class StyleBlock(nn.Module):
    def __init__(self, style_channels: int, out_channels: int) -> None:
        """Add a style vector to the input tensor. See Cellpose.

        Cellpose:
        - https://www.nature.com/articles/s41592-020-01018-x

        Parameters
        ----------
            style_channels : int
                Number of style vector channels.
            out_channels : int
                Number of output channels.
        """
        super().__init__()
        self.out_channels = out_channels
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, x: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """Forward style block."""
        style_feat = self.full(style_feat)
        out = x + style_feat.unsqueeze(-1).unsqueeze(-1)

        return out


class StyleReshape(nn.Module):
    def __init__(
        self,
        in_channels: int,
        style_channels: int,
        normalization: str = "bn",
        convolution: str = "conv",
        **kwargs,
    ) -> None:
        """Reshape feature-map into a style-vector. See Cellpose.

        Cellpose:
        - https://www.nature.com/articles/s41592-020-01018-x

        Takes in a feature map (B, C, H, W). Then averages, and normalizes it
        into to a style feature vector.

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            style_channels : int
                Number of style vector channels.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            convolution : str, default="conv"
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"

        Returns
        -------
            torch.Tensor:
                Style vector. Shape: (B, C).
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.out_channels = style_channels

        self.downsample = None
        if in_channels != style_channels:
            self.downsample = ChannelPool(
                in_channels=in_channels,
                out_channels=style_channels,
                convolution=convolution,
                normalization=normalization,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward style-block."""
        if self.downsample is not None:
            x = self.downsample(x)

        style = x.mean((2, 3), keepdim=True)
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5

        return style
