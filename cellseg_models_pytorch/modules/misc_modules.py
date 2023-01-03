import torch
import torch.nn as nn

from .base_modules import Conv, Norm

__all__ = ["ChannelPool", "StyleBlock", "StyleReshape", "LayerScale"]


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False
    ) -> None:
        """Learnable scaling factor for transformer components.

        I.e. this is used to scale the attention matrix and the mlp head
        of the transformer.

        NOTE: Copied from timm vision_transformer.py


        Parameters
        ----------
            dim : int
                The dimensionality of the input.
            init_values : float, default=1e-5
                Initialization values for the learnable weights.
            inplace : bool, default=False
                Flag, whether the scaling is an inplace operation.
        """
        super().__init__()
        self.dim = dim
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the layer scaling."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def extra_repr(self) -> str:
        """Add extra to repr."""
        return f"dim={self.dim}, inplace={self.inplace}"


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
