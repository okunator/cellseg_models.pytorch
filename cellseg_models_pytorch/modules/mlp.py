from typing import Any, Dict

import torch
import torch.nn as nn

from .base_modules import Activation, Norm

__all__ = ["Mlp", "MlpBlock"]


class Mlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio: int = 2,
        activation: str = "star_relu",
        dropout: float = 0.0,
        bias: bool = False,
        out_channels: int = None,
        **act_kwargs
    ) -> None:
        """MLP token mixer.

        - MetaFormer: https://arxiv.org/abs/2210.13452
        - MLP-Mixer: https://arxiv.org/abs/2105.01601

        - Input shape: (B, N, embed_dim)
        - Output shape: (B, seq_len, embed_dim)

        Parameters
        ----------
            in_channels : int
                Number of input features.
            mlp_ratio : int, default=2
                Scaling factor to get the number hidden features from the `in_features`.
            activation : str, default="star_relu"
                The name of the activation function.
            dropout : float, default=0.0
                Dropout ratio.
            bias : bool, default=False
                Flag whether to use bias terms in the nn.Linear modules.
            out_channels : int, optional
                Number of out channels. If None `out_channels = in_channels`
            **act_kwargs:
                Arbitrary key-word arguments for the activation function.
        """
        super().__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = int(mlp_ratio * in_channels)

        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.act = Activation(activation, **act_kwargs)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_channels, self.out_channels, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass of the MLP token mixer."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio: int = 2,
        activation: str = "star_relu",
        activation_kwargs: Dict[str, Any] = None,
        dropout: float = 0.0,
        bias: bool = False,
        normalization: str = "ln",
        norm_kwargs: Dict[str, Any] = None,
    ) -> None:
        """Residual Mlp block.

        I.e. norm -> mlp -> residual

        Parameters
        ----------
            in_channels : int
                Number of input features.
            mlp_ratio : int, default=2
                Scaling factor to get the number hidden features from the `in_features`.
            activation : str, default="star_relu"
                The name of the activation function.
            dropout : float, default=0.0
                Dropout ratio.
            bias : bool, default=False
                Flag whether to use bias terms in the nn.Linear modules.
            normalization : str, default="ln"
                The name of the normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            norm_kwargs : Dict[str, Any], optional
                key-word args for the normalization layer. Ignored if normalization
                is None.
        """
        super().__init__()
        self.norm = Norm(normalization, **norm_kwargs)
        self.mlp = Mlp(
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
            bias=bias,
            **activation_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Metaformer Mlp-block."""
        residual = x

        x = self.norm(x)
        x = self.mlp(x)

        return x + residual
