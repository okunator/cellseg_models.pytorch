import torch
import torch.nn as nn

__all__ = ["SegHead"]


class SegHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 1
    ) -> None:
        """Segmentation head at the end of decoder branches.

        Parameters
        ----------
            in_channels : int
                Number of channels in the input tensor.
            out_channels : int
                Number of channels in the output tensor.
            kernel_size : int, default=1
                Kernel size for the conv operation.

        """
        super().__init__()
        self.n_classes = out_channels

        if kernel_size != 1:
            self.head = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.head = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation head."""
        return self.head(x)
