import torch
import torch.nn as nn

from .act import ACT_LOOKUP
from .conv import CONV_LOOKUP
from .norm import NORM_LOOKUP
from .upsample import UP_LOOKUP

__all__ = ["Activation", "Norm", "Up", "Conv", "Identity"]


class Identity(nn.Module):
    def __init__(self, in_channels: int = None, *args, **kwargs):
        """Identity operator that is argument-insensitive.

        The forward method can take in multiple arguments returning only the
        first one.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of idenStity operator."""
        return input


class Activation(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Activation wrapper class.

        Parameters:
        -----------
            name : str
                Name of the activation method.

        Raises
        ------
            ValueError: if the activation method name is illegal.
        """
        super().__init__()

        allowed = list(ACT_LOOKUP.keys()) + [None]
        if name not in allowed:
            raise ValueError(
                f"Illegal activation method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name is not None:
            try:
                self.act = ACT_LOOKUP[name](**kwargs, inplace=True)
            except Exception:
                self.act = ACT_LOOKUP[name](**kwargs)
        else:
            self.act = Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the activation function."""
        return self.act(x)


class Norm(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Normalize wrapper class.

        Parameters:
        -----------
            name : str
                Name of the normalization method.

        Raises
        ------
            ValueError: if the normalization method name is illegal.
        """
        super().__init__()

        allowed = list(NORM_LOOKUP.keys()) + [None]
        if name not in allowed:
            raise ValueError(
                f"Illegal normalization method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name is not None:
            self.norm = NORM_LOOKUP[name](**kwargs)
        else:
            self.norm = Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the norm function."""
        return self.norm(x)


class Up(nn.Module):
    def __init__(self, name: str, scale_factor: int = 2, **kwargs) -> None:
        """Upsample wrapper class.

        Parameters:
        -----------
            name : str
                Name of the upsampling method.
            scale_factor : int, default=2
                Upsampling scale factor. scale_factor*(H, W)

        Raises
        ------
            ValueError: if the upsampling method name is illegal.
        """
        super().__init__()

        allowed = list(UP_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal upsampling method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name in ("bilinear", "bicubic"):
            kwargs["mode"] = name
            kwargs["align_corners"] = True

        kwargs["scale_factor"] = scale_factor
        self.up = UP_LOOKUP[name](**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the upsampling function."""
        return self.up(x)


class Conv(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Convolution wrapper class.

        Parameters:
        -----------
            name : str
                Name of the convolution method.

        Raises
        ------
            ValueError: if the convolution method name is illegal.
        """
        super().__init__()

        allowed = list(CONV_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal convolution method given. Allowed: {allowed}. Got: '{name}'"
            )

        self.conv = CONV_LOOKUP[name](**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the convolution function."""
        return self.conv(x)
