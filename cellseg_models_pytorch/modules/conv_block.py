import torch
import torch.nn as nn

from .conv_base import (
    BasicConv,
    BottleneckConv,
    DepthWiseSeparableConv,
    FusedMobileInvertedConv,
    HoverNetDenseConv,
    InvertedBottleneckConv,
)
from .misc_modules import ChannelPool, StyleBlock

__all__ = ["ConvBlock"]

CONVBLOCK_LOOKUP = {
    "basic": BasicConv,
    "bottleneck": BottleneckConv,
    "dws": DepthWiseSeparableConv,
    "mbconv": InvertedBottleneckConv,
    "fmbconv": FusedMobileInvertedConv,
    "hover_dense": HoverNetDenseConv,
}


class ShortSkipMixIn:
    """Add short skip functionality.

    - Dense short skip forward method
    - Residual short skip forward method
    - Forward method w/o short skip.
    """

    def forward_basic(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a generic forward pass without short skip connection."""
        if self.block.preactivate:
            out = self.block.forward_features_preact(x)
        else:
            out = self.block.forward_features(x)

        return out

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a generic forward pass with residual short skip."""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.block.preactivate:
            out = self.block.forward_features_preact(x)
        else:
            out = self.block.forward_features(x)

        out = out + identity

        return out

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a generic forward pass with dense short skip connection."""
        prev_features = [x] if isinstance(x, torch.Tensor) else x
        x = torch.cat(prev_features, dim=1)

        if self.block.preactivate:
            out = self.block.forward_features_preact(x)
        else:
            out = self.block.forward_features(x)

        return out


class ConvBlock(nn.Module, ShortSkipMixIn):
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        style_channels: int = None,
        short_skip: str = "residual",
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
        attention: str = None,
        preattend: bool = False,
        use_style: bool = True,
        **kwargs,
    ) -> None:
        """Wrap different conv-blocks under one generic conv-block.

        Adds one channel pooling block if `short_skip` == "residual".

        Optional:
            - add a style vector to the output at the end of the block (Cellpose).

        Parameters
        ----------
            name : str
                The name of the conv-block. One of: "basic". "mbconv", "fmbconv" "dws",
                "bottleneck".
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
            short_skip : str, default="residual"
                The name of the short skip method. One of: "residual", "dense", "basic"
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
            activation : str, default="relu"
                Activation method.
                One of: "mish", "swish", "relu", "relu6", "rrelu", "selu",
                "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
            convolution : str, default="conv"
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            preactivate : bool, default=False
                If True, normalization will be applied before convolution.
            kernel_size : int, default=3
                The size of the convolution kernel.
            groups : int, default=1
                Number of groups the kernels are divided into. If `groups == 1`
                normal convolution is applied. If `groups = in_channels`
                depthwise convolution is applied.
            bias : bool, default=True,
                Include bias term in the convolution block. Only used for `BaasicConv`.
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
            use_style : bool, default=False
                If True and `style_channels` is not None, adds a style vec to output.

        Raises
        ------
            ValueError:
                - If illegal `name`is given as input argument.
                - If illegal `short_skip` is given as input argument.
        """
        super().__init__()
        self.short_skip = short_skip

        allowed = list(CONVBLOCK_LOOKUP.keys())
        if name not in allowed:
            raise ValueError(
                f"Illegal convblock name given. Got '{name}'. Allowed: {allowed}"
            )

        allowed = ("residual", "dense", "basic")
        if short_skip not in allowed:
            raise ValueError(
                f"Illegal `short_skip` given. Got: '{short_skip}'. Allowed: {allowed}."
            )

        try:
            self.block = CONVBLOCK_LOOKUP[name](
                in_channels=in_channels,
                out_channels=out_channels,
                same_padding=same_padding,
                normalization=normalization,
                activation=activation,
                convolution=convolution,
                preactivate=preactivate,
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
                attention=attention,
                preattend=preattend,
                **kwargs,
            )
        except Exception as e:
            raise Exception(
                "Encountered an error when trying to init ConvBlock module: "
                f"ConvBlock(name='{name}'): {e.__class__.__name__}: {e}"
            )

        self.downsample = None
        if short_skip == "residual" and in_channels != self.out_channels:
            self.downsample = ChannelPool(
                in_channels=in_channels,
                out_channels=self.out_channels,
                convolution=convolution,
                normalization=normalization,
            )

        self.add_style = None
        if style_channels is not None and use_style:
            self.add_style = StyleBlock(style_channels, out_channels)

    @property
    def out_channels(self) -> int:
        """Set out_channels."""
        return self.block.out_channels

    def forward(self, x: torch.Tensor, style: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the conv-block."""
        if self.short_skip == "residual":
            x = self.forward_residual(x)
        elif self.short_skip == "dense":
            x = self.forward_dense(x)
        else:
            x = self.forward_basic(x)

        if self.add_style is not None:
            x = self.add_style(x, style)

        return x
