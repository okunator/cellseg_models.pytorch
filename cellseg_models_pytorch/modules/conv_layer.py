from typing import List, Tuple

import torch
import torch.nn as nn

from .conv_block import ConvBlock
from .misc_modules import ChannelPool

__all__ = ["ConvLayer"]


class ConvLayer(nn.ModuleDict):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int = 2,
        layer_residual: bool = False,
        short_skip: str = "residual",
        style_channels: int = None,
        expand_ratios: Tuple[float, float] = (1.0, 1.0),
        block_types: Tuple[str, ...] = ("basic", "basic"),
        normalizations: Tuple[str, ...] = ("bn", "bn"),
        activations: Tuple[str, ...] = ("relu", "relu"),
        convolutions: Tuple[str, ...] = ("conv", "conv"),
        kernel_sizes: Tuple[int, ...] = (3, 3),
        groups: Tuple[int, ...] = (1, 1),
        biases: Tuple[bool, ...] = (True, True),
        preactivates: Tuple[bool, ...] = (False, False),
        attentions: Tuple[str, ...] = (None, None),
        preattends: Tuple[bool, ...] = (False, False),
        use_styles: Tuple[bool, ...] = (False, False),
        **kwargs,
    ) -> None:
        """Stack conv-blocks in a ModuleDict to compose a full layer.

        Optional:
            - add a style vector to the output at the end of each conv block (Cellpose)

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_blocks: Number of ConvBlocks used in this layer.
                Defaults to 2.
            layer_residual: Apply a layer level residual skip. I.e x + layer(x)
                Defaults to False.
            style_channels:
                Number of style vector channels. If None, style vectors are ignored.
                Defaults to None.
            short_skip:
                The name of the short skip method. One of: "residual", "dense", "basic"
                Defaults to "residual".
            expand_ratios (Tuple[float, ...]):
                Expansion/Squeeze ratios for the out channels of each conv block.
                Defaults to (1.0, 1.0).
            block_types:
                The name of the conv-blocks. Length of the tuple has to equal `n_blocks`
                One of: "basic". "mbconv", "fmbconv" "dws", "bottleneck".
                Defaults to ("basic", "basic").
            normalizations:
                Normalization methods. One of: "bn", "bcn", "gn", "in", "ln", "lrn"
                Defaults to ("bn", "bn").
            activations:
                Activation methods. One of: "mish", "swish", "relu", "relu6", "rrelu",
                "selu", "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
                Defaults to ("relu", "relu").
            convolutions:
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
                Defaults to ("conv", "conv").
            preactivates: Pre-activations flags for the conv-blocks.
                Defaults to (False, False).
            kernel_sizes: The size of the convolution kernels in each conv block.
                Defaults to (3, 3).
            groups (int):
                Number of groups for the kernels in each convolution blocks.
                Defaults to (1, 1).
            biases (bool): Include bias terms in the convolution blocks.
                Defaults to (True, True).
            attentions:
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
                Defaults to (None, None).
            preattends:
                If True, Attention is applied at the beginning of forward pass.
                Defaults to (False, False).
            use_styles (bool):
                If True and `style_channels` is not None, adds a style vec to output.
                Defaults to (False, False).

        Raises:
            ValueError:
                If lengths of the tuple arguments are not equal to `n_blocks`.
        """
        super().__init__()
        self.layer_residual = layer_residual
        self.short_skip = short_skip
        self.in_channels = in_channels

        illegal_args = [
            (k, a)
            for k, a in locals().items()
            if isinstance(a, tuple) and len(a) != n_blocks
        ]

        if illegal_args:
            raise ValueError(
                f"""
                All the tuple-arg lengths need to be equal to `n_blocks`={n_blocks}.
                Illegal args: {illegal_args}"""
            )

        blocks = list(range(n_blocks))
        for i in blocks:
            out = int(out_channels * expand_ratios[i])

            conv_block = ConvBlock(
                name=block_types[i],
                in_channels=in_channels,
                out_channels=out,
                style_channels=style_channels,
                short_skip=short_skip,
                kernel_size=kernel_sizes[i],
                groups=groups[i],
                bias=biases[i],
                normalization=normalizations[i],
                convolution=convolutions[i],
                activation=activations[i],
                attention=attentions[i],
                preactivate=preactivates[i],
                preattend=preattends[i],
                use_style=use_styles[i],
                **kwargs,
            )
            self.add_module(f"{short_skip}_{block_types[i]}_{i + 1}", conv_block)

            if short_skip == "dense":
                in_channels += conv_block.out_channels
            else:
                in_channels = conv_block.out_channels

        self.out_channels = conv_block.out_channels

        if short_skip == "dense":
            self.transition = ConvBlock(
                name="basic",
                in_channels=in_channels,
                short_skip="basic",
                out_channels=out_channels,
                same_padding=False,
                bias=False,
                kernel_size=1,
                convolution=conv_block.block.conv_choice,
                normalization=normalizations[-1],
                activation=activations[-1],
                preactivate=preactivates[-1],
            )
            self.out_channels = self.transition.out_channels

        self.downsample = None
        if layer_residual and self.in_channels != self.out_channels:
            self.downsample = ChannelPool(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                convolution=convolutions[-1],
                normalization=normalizations[-1],
            )

    def forward_features_dense(
        self, init_features: List[torch.Tensor], style: torch.Tensor = None
    ) -> torch.Tensor:
        """Dense forward pass."""
        features = [init_features]
        for name, conv_block in self.items():
            if name not in ("transition", "downsample"):
                new_features = conv_block(features, style)
                features.append(new_features)

        x = torch.cat(features, 1)
        x = self.transition(x)

        return x

    def forward_features(
        self, x: torch.Tensor, style: torch.Tensor = None
    ) -> torch.Tensor:
        """Regular forward pass."""
        for name, conv_block in self.items():
            if name != "downsample":
                x = conv_block(x, style)

        return x

    def forward(self, x: torch.Tensor, style: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the conv-layer."""
        if self.layer_residual:
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)

        if self.short_skip == "dense":
            x = self.forward_features_dense(x, style)
        else:
            x = self.forward_features(x, style)

        if self.layer_residual:
            x = x + identity

        return x
