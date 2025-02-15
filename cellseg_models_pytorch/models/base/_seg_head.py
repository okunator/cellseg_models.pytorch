import torch
import torch.nn as nn

from cellseg_models_pytorch.modules import ConvBlock

__all__ = ["SegHead"]


class SegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = False,
        excitation_channels: int = None,
    ) -> None:
        """Segmentation head at the end of decoder branches.

        Parameters:
            in_channels (int):
                Number of channels in the input tensor.
            out_channels (int):
                Number of channels in the output tensor.
            kernel_size (int, default=1):
                Kernel size for the conv operation.
            bias (bool, default=False):
                If True, add a bias term to the conv operation.
            excitation_channels (int, default=None):
                Number of channels in an optional excitation conv layer before the
                output head.

        """
        super().__init__()
        self.n_classes = out_channels

        self.excite = None
        if excitation_channels is not None:
            self.excite = ConvBlock(
                name="basic",
                in_channels=in_channels,
                out_channels=excitation_channels,
                short_skip="basic",
                kernel_size=3,
                normalization=None,
                activation="relu",
                convolution="conv",
                preactivate=False,
                bias=False,
            )
            in_channels = self.excite.out_channels

        if kernel_size != 1:
            self.head = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.head = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation head."""
        if self.excite is not None:
            x = self.excite(x)
        return self.head(x)
