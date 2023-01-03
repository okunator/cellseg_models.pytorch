from typing import Any, Dict

import torch
import torch.nn as nn

from .attention_modules import MSCA
from .base_modules import Activation, Conv, Identity, Norm
from .mlp import Mlp
from .self_attention_modules import SelfAttention

__all__ = ["MSCAN", "Pooling", "TokenMixer", "TokenMixerBlock"]


class MSCAN(nn.Module):
    def __init__(
        self, in_channels: int, conv: str = "conv", activation: str = "gelu", **kwargs
    ) -> None:
        """Create MSCAN spatial attention module.

        Parameters
        ----------
            conv : str, default="conv"
                Convolution layer type.
            activation : str, default="relu"
                Activation layer after squeeze.
        """
        super().__init__()
        self.in_channels = in_channels
        self.proj_1 = Conv(
            conv, in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )
        self.activation = Activation(activation)
        self.spatial_gating_unit = MSCA(in_channels)
        self.proj_2 = Conv(
            conv, in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MSCAN attention."""
        residual = x

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)

        return x + residual


class Pooling(nn.Module):
    def __init__(self, kernel_size: int = 3, **kwargs):
        """Init a basic pooling module.

        - PoolFormer: https://arxiv.org/abs/2111.11418

        Parameters
        ----------
            kernel_size_size : int, default=3
                The pooling kernel_size.
        """
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pool = nn.AvgPool2d(
            kernel_size, stride=1, padding=padding, count_include_pad=False
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the pooling module."""
        y = self.pool(x)

        return y - x


class TokenMixer(nn.Module):
    def __init__(self, name: str, mixer_kwargs: Dict[str, Any], **kwargs) -> None:
        """Token mixer wrapper class.

        Parameters
        ----------
            name : str
                Name of the token-mixer. Allowed: "pool", "self-attention", "mscan",
                "identity", "mlp"
        """
        super().__init__()
        allowed = list(MIXER_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal token mixer given. Allowed: {allowed}. Got: '{name}'"
            )

        try:
            self.mixer = MIXER_LOOKUP[name](**mixer_kwargs)
        except Exception as e:
            raise Exception(
                "Encountered an error when trying to init token mixer: "
                f"TokenMixer(name='{name}'): {e.__class__.__name__}: {e}"
            )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of a token mixer."""
        return self.mixer(x, **kwargs)


class TokenMixerBlock(nn.Module):
    def __init__(
        self,
        token_mixer: str,
        normalization: str,
        residual: bool = True,
        norm_kwargs: Dict[str, Any] = None,
        mixer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Token mixer block.

        I.e. norm(x) -> tokenmixer(x) -> residual -> (reshape for MLP)

        Parameters
        ----------
            token_mixer : str
                Name of the token mixer. Allowed: "pool", "self-attention", "mscan",
                "identity", "mlp".
            normalization : str
                Name of the normalization method. Allowed: "bn", "bcn", "gn", "in",
                "ln", None.
            residual : bool, default=True
                Flag, whether to use a residual connection at the end of the mixer.
            norm_kwargs : Dict[str, Any], optional
                Arbitrary key-word arguments for the normalization method.
            mixer_kwargs : Dict[str, Any], optional
                Arbitrary key-word arguments for the token mixer module.
        """
        super().__init__()
        self.residual = residual
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        mixer_kwargs = mixer_kwargs if mixer_kwargs is not None else {}
        self.norm = Norm(normalization, **norm_kwargs)
        self.token_mixer = TokenMixer(token_mixer, mixer_kwargs)
        self.reshape = RESHAPE_LOOKUP[token_mixer]  # (B, C, H, W) -> (B, N, C)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the token-mixer block.

        Parameters
        ----------
            x : torch.Tensor
                Input features of shape (B, C, H, W) or (B, N, C).
            **kwargs:
                Arbitrary key-word arguments such e.g. `context` for cross-attn.
        """
        if self.reshape:
            B, C, H, W = x.shape

        if self.residual:
            residual = x

        x = self.norm(x)
        x = self.token_mixer(x, **kwargs)

        if self.residual:
            x = x + residual

        if self.reshape:
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x


MIXER_LOOKUP = {
    "pool": Pooling,
    "mscan": MSCAN,
    "mlp": Mlp,
    "self-attention": SelfAttention,
    "identity": Identity,
}

RESHAPE_LOOKUP = {
    "pool": True,
    "mscan": True,
    "mlp": False,
    "self-attention": False,
    "identity": False,
}
