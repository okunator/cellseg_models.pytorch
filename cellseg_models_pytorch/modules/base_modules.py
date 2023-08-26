import torch
import torch.nn as nn

from .act import ACT_LOOKUP
from .conv import CONV_LOOKUP
from .norm import NORM_LOOKUP
from .self_attention import SELFATT_LOOKUP
from .upsample import UP_LOOKUP

__all__ = ["Activation", "Norm", "Up", "Conv", "Identity", "MultiHeadSelfAttention"]


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
                try:
                    self.act = ACT_LOOKUP[name](**kwargs)
                except Exception as e:
                    raise Exception(
                        "Encountered an error when trying to init activation function: "
                        f"Activation(name='{name}'): {e.__class__.__name__}: {e}"
                    )
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
            try:
                self.norm = NORM_LOOKUP[name](**kwargs)
            except Exception as e:
                raise Exception(
                    "Encountered an error when trying to init normalization function: "
                    f"Norm(name='{name}'): {e.__class__.__name__}: {e}"
                )
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
                Name of the upsampling method. One of: 'bilinear', 'bicubic',
                'fixed-unpool', 'conv_transpose', 'nearest'
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

        if name == "conv_transpose":
            kwargs["kernel_size"] = scale_factor
            kwargs["stride"] = scale_factor
            kwargs["padding"] = 0
            kwargs["output_padding"] = 0
        else:
            kwargs["scale_factor"] = scale_factor
            kwargs.pop("in_channels", None)
            kwargs.pop("out_channels", None)

        if scale_factor == 1:
            self.up = Identity()
        else:
            try:
                self.up = UP_LOOKUP[name](**kwargs)
            except Exception as e:
                raise Exception(
                    "Encountered an error when trying to init upsampling function: "
                    f"Up(name='{name}'): {e.__class__.__name__}: {e}"
                )

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

        try:
            self.conv = CONV_LOOKUP[name](**kwargs)
        except Exception as e:
            raise Exception(
                "Encountered an error when trying to init convolution function: "
                f"Conv(name='{name}'): {e.__class__.__name__}: {e}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the convolution function."""
        return self.conv(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Multi-head self-attention wrapper class.

        Parameters:
        -----------
            name : str
                Name of the mhsa method.

        Raises
        ------
            ValueError: if the mhsa method name is illegal.
        """
        super().__init__()

        allowed = list(SELFATT_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                "Illegal multi-head attention method given. "
                f"Allowed: {allowed}. Got: '{name}'"
            )

        try:
            self.att = SELFATT_LOOKUP[name](**kwargs)
        except Exception as e:
            raise Exception(
                "Encountered an error when trying to init convolution function: "
                f"MultiHeadSelfAttention(name='{name}'): {e.__class__.__name__}: {e}"
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for the convolution function."""
        return self.att(query, key, value, **kwargs)
