from typing import List, Tuple

import torch
import torch.nn as nn

from .base_modules import Activation, Conv, Norm
from .conv_block import ConvBlock

__all__ = ["ConvLayer"]


class ConvLayer(nn.ModuleDict):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        short_skip: str = "residual",
        block_types: Tuple[str, ...] = ("basic", "basic"),
        normalizations: Tuple[str, ...] = ("bn", "bn"),
        activations: Tuple[str, ...] = ("relu", "relu"),
        convolutions: Tuple[str, ...] = ("conv", "conv"),
        kernel_sizes: Tuple[int, ...] = (3, 3),
        preactivates: Tuple[bool, ...] = (False, False),
        attentions: Tuple[str, ...] = (None, None),
        preattends: Tuple[bool, ...] = (False, False),
        **kwargs,
    ) -> None:
        """Stack conv-blocks in a ModuleDict to compose a full layer.

        Parameters
        ----------
            n_blocks : int
                Number of ConvBlocks used in this layer.
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            short_skip : str, default="residual"
                The name of the short skip method. One of: "residual", "dense", "basic"
            block_types : Tuple[str, ...], default=("basic", "basic")
                The name of the conv-blocks. Length of the tuple has to equal `n_blocks`
                One of: "basic". "mbconv", "fmbconv" "dws", "bottleneck".
            normalizations : Tuple[str, ...], default=("bn", "bn"):
                Normalization methods. One of: "bn", "bcn", "gn", "in", "ln", "lrn"
            activations : Tuple[str, ...], default=("relu", "relu")
                Activation methods. One of: "mish", "swish", "relu", "relu6", "rrelu",
                "selu", "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
            convolutions : Tuple[str, ...], default=("conv", "conv")
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            preactivates : Tuple[bool, ...], default=(False, False)
                Pre-activations flags for the conv-blocks.
            kernel_sizes : Tuple[int, ...], default=(3, 3)
                The size of the convolution kernels.
            attentions : Tuple[str, ...], default=(None, None)
                Attention method. One of: "se", "scse", "gc", "eca", None
            preattends : Tuple[bool, ...], default=(False, False)
                If True, Attention is applied at the beginning of forward pass.

        Raises
        ------
            ValueError:
                If lengths of the tuple arguments are not equal to `n_blocks`.
        """
        super().__init__()
        self.short_skip = short_skip

        if not all(
            [len(a) == n_blocks for a in locals().values() if isinstance(a, tuple)]
        ):
            raise ValueError(
                f"All the tuple-arg lengths need to be equal to `n_blocks`={n_blocks}."
            )

        blocks = list(range(n_blocks))
        for i in blocks:
            conv_block = ConvBlock(
                name=block_types[i],
                in_channels=in_channels,
                out_channels=out_channels,
                short_skip=short_skip,
                kernel_size=kernel_sizes[i],
                normalization=normalizations[i],
                convolution=convolutions[i],
                activation=activations[i],
                attention=attentions[i],
                preactivate=preactivates[i],
                preattend=preattends[i],
            )
            self.add_module(f"{short_skip}_{block_types[i]}_{i + 1}", conv_block)

            if short_skip == "dense":
                in_channels += conv_block.out_channels
            else:
                in_channels = conv_block.out_channels

        if short_skip == "dense":
            self.transition = nn.Sequential(
                Conv(
                    conv_block.block.conv_choice,
                    in_channels=in_channels,
                    bias=False,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                ),
                Norm(normalizations[-1], num_features=out_channels),
                Activation(activations[-1]),
            )

        self.out_channels = in_channels

    def forward_features_dense(self, init_features: List[torch.Tensor]) -> torch.Tensor:
        """Dense forward pass."""
        features = [init_features]
        for name, conv_block in self.items():
            if name != "transition":
                new_features = conv_block(features)
                features.append(new_features)

        x = torch.cat(features, 1)
        x = self.transition(x)

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Regular forward pass."""
        for _, conv_block in self.items():
            x = conv_block(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the conv-layer."""
        if self.short_skip == "dense":
            x = self.forward_features_dense(x)
        else:
            x = self.forward_features(x)

        return x
