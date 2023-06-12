from typing import Any, Dict

import torch
import torch.nn as nn

from .base_modules import Activation, Norm

__all__ = ["Mlp", "ConvMlp", "MlpBlock"]


class Mlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio: int = 2,
        activation: str = "star_relu",
        dropout: float = 0.0,
        bias: bool = False,
        out_channels: int = None,
        act_kwargs: Dict[str, Any] = None,
        **kwargs,
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
                Scaling factor to get the number hidden features from the `in_channels`.
            activation : str, default="star_relu"
                The name of the activation function.
            dropout : float, default=0.0
                Dropout ratio.
            bias : bool, default=False
                Flag whether to use bias terms in the nn.Linear modules.
            out_channels : int, optional
                Number of out channels. If None `out_channels = in_channels`
            act_kwargs : Dict[str, Any], optional
                Arbitrary key-word arguments for the activation function.
        """
        super().__init__()
        act_kwargs = act_kwargs if act_kwargs is not None else {}
        self.out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = int(mlp_ratio * in_channels)
        act_kwargs["dim_in"] = hidden_channels
        act_kwargs["dim_out"] = hidden_channels

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


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_ratio: int = 2,
        activation: str = "star_relu",
        dropout: float = 0.0,
        bias: bool = False,
        out_channels: int = None,
        act_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Mlp layer implemented with dws convolution.

        Input shape: (B, in_channels, H, W).
        Output shape: (B, out_channels, H, W).

        Parameters
        ----------
            in_channels : int
                Number of input features.
            mlp_ratio : int, default=2
                Scaling factor to get the number hidden features from the `in_channels`.
            activation : str, default="star_relu"
                The name of the activation function.
            dropout : float, default=0.0
                Dropout ratio.
            bias : bool, default=False
                Flag whether to use bias terms in the nn.Linear modules.
            out_channels : int, optional
                Number of out channels. If None `out_channels = in_channels`
            act_kwargs : Dict[str, Any], optional
                Arbitrary key-word arguments for the activation function.
        """
        super().__init__()
        act_kwargs = act_kwargs if act_kwargs is not None else {}
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = int(mlp_ratio * in_channels)
        self.fc1 = nn.Conv2d(in_channels, self.hidden_channels, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, 3, 1, 1, bias=bias, groups=in_channels
        )
        self.act = Activation(activation, **act_kwargs)
        self.fc2 = nn.Conv2d(self.hidden_channels, self.out_channels, 1, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of conv-mlp."""
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class MlpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mlp_type: str = "linear",
        mlp_ratio: int = 2,
        activation: str = "star_relu",
        act_kwargs: Dict[str, Any] = None,
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
            mlp_type : str, default="linear"
                Flag for either nn.Linear or nn.Conv2d mlp-layer.
                One of "conv", "linear".
            mlp_ratio : int, default=2
                Scaling factor to get the number hidden features from the `in_channels`.
            activation : str, default="star_relu"
                The name of the activation function.
            act_kwargs : Dict[str, Any], optional
                key-word args for the activation module.
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
        allowed = ("conv", "linear")
        if mlp_type not in allowed:
            raise ValueError(
                f"Illegal `mlp_type` given. Got: {mlp_type}. Allowed: {allowed}."
            )

        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        act_kwargs = act_kwargs if act_kwargs is not None else {}
        self.norm = Norm(normalization, **norm_kwargs)
        MlpHead = Mlp if mlp_type == "linear" else ConvMlp

        self.mlp = MlpHead(
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
            bias=bias,
            act_kwargs=act_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Metaformer Mlp-block."""
        residual = x

        x = self.norm(x)
        x = self.mlp(x)

        return x + residual
