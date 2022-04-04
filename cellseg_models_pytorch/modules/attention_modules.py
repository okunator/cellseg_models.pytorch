"""
Spatial attention modules for CNNs.

Most of these are simplified and adapted from:
https://github.com/rwightman/pytorch-image-models

License:

Copyright 2019 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers.helpers import make_divisible

from .base_modules import Activation, Conv, Identity, Norm

__all__ = [
    "Attention",
    "SqueezeAndExcite",
    "SCSqueezeAndExcite",
    "ECA",
    "GlobalContext",
]


class SqueezeAndExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float = 0.25,
        conv: str = "conv",
        activation: str = "relu",
        gate_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        """Squeeze-and-Excitation block.

        https://arxiv.org/abs/1709.01507

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            squeeze_ratio : float, default=0.25
                Ratio of squeeze.
            conv : str, default="conv"
                Convolution layer type.
            activation : str, default="relu"
                Activation layer after squeeze.
            gate_activation : str, default="sigmoid"
                Attention gate function.
        """
        super().__init__()

        squeeze_channels = round(in_channels * squeeze_ratio)

        if squeeze_channels < 2:
            squeeze_channels = 2

        # squeeze channel pooling
        self.conv_squeeze = Conv(
            conv,
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.act = Activation(activation)

        # excite channel pooling
        self.conv_excite = Conv(
            conv,
            in_channels=squeeze_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.gate = Activation(gate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SE-forward pass."""
        # squeeze
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_squeeze(x_se)
        x_se = self.act(x_se)

        # excite
        x_se = self.conv_excite(x_se)

        return x * self.gate(x_se)


class SCSqueezeAndExcite(SqueezeAndExcite):
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float = 0.25,
        conv: str = "conv",
        activation: str = "relu",
        gate_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        """Spatial and Channel Squeeze & Excitation.

        https://arxiv.org/abs/1803.02579

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            squeeze_ratio : float, default=0.25
                Ratio of squeeze.
            conv : str, default="conv"
                Convolution layer type.
            activation : str, default="relu"
                Activation layer after squeeze.
            gate_activation : str, default="sigmoid"
                Attention gate function.
        """
        super().__init__(
            in_channels=in_channels,
            squeeze_ratio=squeeze_ratio,
            conv=conv,
            activation=activation,
            gate_activation=gate_activation,
        )

        # self.conv_squeeze2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_squeeze2 = Conv(
            conv, in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )
        self.gate2 = Activation(gate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SCSE forward pass."""
        # Channel-attention
        x_ce = x.mean((2, 3), keepdim=True)
        x_ce = self.conv_squeeze(x_ce)
        x_ce = self.act(x_ce)
        x_ce = self.conv_excite(x_ce)

        # Spatial attention
        x_se = self.conv_squeeze2(x)

        return x * self.gate(x_ce) + x * self.gate2(x_se)


class ECA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        beta: int = 1,
        gamma: int = 2,
        gate_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        """Efficient Channel Attention (ECA).

        https://arxiv.org/abs/1910.03151

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            beta : int, default=1
                Coefficient used to compute the kernel size adaptively.
            gamma : int, default=2
                Coefficient used to compute the kernel size adaptively.
            gate_activation : str, default="sigmoid"
                Attention gate function.
        """
        super().__init__()

        # Compute the adaptive kernel size
        t = int(abs(math.log(in_channels, 2) + beta) / gamma)
        kernel_size = max(t if t % 2 else t + 1, 3)

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False
        )

        self.gate = Activation(gate_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ECA Forward pass."""
        B = x.shape[0]

        y = x.mean((2, 3)).view(B, 1, -1)
        y = self.conv(y)
        y = self.gate(y).view(B, -1, 1, 1)

        return x * y.expand_as(x)


class GlobalContext(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float = 0.125,
        conv: str = "conv",
        activation: str = "relu",
        **kwargs,
    ) -> None:
        """Global context attention block.

        https://arxiv.org/abs/1904.11492

        NOTE: Only the (attn + add)-fusion (the best) variant implemented.

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            squeeze_ratio : float, default=0.25
                Ratio of squeeze.
            conv : str, default="conv"
                Convolution layer type.
            activation : str, default="relu"
                Activation layer after squeeze.
        """
        super().__init__()

        self.conv_attn = Conv(
            conv,
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        squeeze_channels = make_divisible(in_channels * squeeze_ratio)

        self.conv_squeeze = Conv(
            conv,
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.norm = Norm("ln", num_features=squeeze_channels)
        self.act = Activation(activation)
        self.conv_excite = Conv(
            conv,
            in_channels=squeeze_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Init conv kernel weights."""
        nn.init.kaiming_normal_(
            self.conv_attn.conv.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.conv_excite.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of global context block."""
        B, C, H, W = x.shape

        # Query independent global context attention
        attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
        attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
        context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
        context = context.view(B, C, 1, 1)

        # squeeze and excite for the global context.
        context = self.conv_squeeze(context)
        context = self.norm(context)
        context = self.act(context)
        context = self.conv_excite(context)

        # fuse (add)
        x = x + context

        return x


ATT_LOOKUP = {
    "se": SqueezeAndExcite,
    "scse": SCSqueezeAndExcite,
    "eca": ECA,
    "gc": GlobalContext,
}


class Attention(nn.Module):
    def __init__(self, name: str, **kwargs) -> None:
        """Attention wrapper class.

        Parameters:
        -----------
            name : str
                Name of the attention method.
        """
        super().__init__()

        allowed = list(ATT_LOOKUP.keys()) + [None]
        if name not in allowed:
            raise ValueError(
                f"Illegal attention method given. Allowed: {allowed}. Got: '{name}'"
            )

        if name is not None:
            self.att = ATT_LOOKUP[name](**kwargs)
        else:
            self.att = Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the attention method."""
        return self.att(x)
