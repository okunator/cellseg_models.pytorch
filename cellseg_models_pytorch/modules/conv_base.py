import torch
import torch.nn as nn
from timm.layers import make_divisible

from .attention_modules import Attention
from .base_modules import Activation, Conv, Norm

__all__ = [
    "BasicConv",
    "BottleneckConv",
    "DepthWiseSeparableConv",
    "InvertedBottleneckConv",
    "FusedMobileInvertedConv",
    "HoverNetDenseConv",
]


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size=3,
        groups: int = 1,
        bias: bool = False,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Conv-block (basic) parent class.

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
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
            bias : bool, default=False,
                Include bias term in the convolution.
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()
        self.conv_choice = convolution
        self.out_channels = out_channels
        self.preattend = preattend
        self.preactivate = preactivate

        # set norm channel number for preactivation or normal
        norm_channels = in_channels if preactivate else self.out_channels

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0

        self.conv = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            bias=bias,
        )

        self.norm = Norm(normalization, num_features=norm_channels)
        self.act = Activation(activation)

        # set attention channels
        att_channels = in_channels if preattend else self.out_channels
        self.att = Attention(attention, in_channels=att_channels)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        if not self.preattend:
            x = self.att(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        if self.preattend:
            x = self.att(x)

        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)

        if not self.preattend:
            x = self.att(x)

        return x


class BottleneckConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        base_width: int = 64,
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = False,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Bottleneck conv block parent-class.

        Res-Net: Deep residual learning for image recognition:
            - https://arxiv.org/abs/1512.03385

        Preact-ResNet: Identity Mappings in Deep Residual Networks:
            - https://arxiv.org/abs/1603.05027

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            expand_ratio : float, default=4.0
                The ratio of channel expansion in the bottleneck.
            base_width : int, default=64
                The minimum width for the conv x channels in this block.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
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
            bias : bool, default=False,
                Include bias term in the convolution.
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()
        self.conv_choice = convolution
        self.expansion = int(expand_ratio)
        self.preattend = preattend
        self.preactivate = preactivate

        width = int(out_channels * (base_width / 64.0)) * groups
        self.out_channels = out_channels * self.expansion

        # set attention channels
        att_channels = in_channels if preattend else self.out_channels
        self.att = Attention(attention, in_channels=att_channels)

        self.conv1 = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            bias=bias,
            padding=0,
        )

        norm_channels = in_channels if preactivate else width
        self.norm1 = Norm(normalization, num_features=norm_channels)
        self.act1 = Activation(activation)

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv2 = Conv(
            name=self.conv_choice,
            in_channels=width,
            out_channels=width,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            bias=bias,
        )

        self.norm2 = Norm(normalization, num_features=width)
        self.act2 = Activation(activation)

        self.conv3 = Conv(
            name=self.conv_choice,
            in_channels=width,
            bias=bias,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
        )

        norm_channels = width if preactivate else self.out_channels
        self.norm3 = Norm(normalization, num_features=norm_channels)

        self.act3 = Activation(activation)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.act3(x)

        if not self.preattend:
            x = self.att(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        if self.preattend:
            x = self.att(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        x = self.norm3(x)
        x = self.conv3(x)

        x = self.act3(x)

        if not self.preattend:
            x = self.att(x)

        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size: int = 3,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Depthwise separable conv block parent class.

        MobileNets:
        Efficient Convolutional Neural Networks for Mobile Vision Applications:
            - https://arxiv.org/abs/1704.04861

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
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
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()
        self.conv_choice = convolution
        self.out_channels = out_channels
        self.preattend = preattend
        self.preactivate = preactivate

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0

        self.depth_conv = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
        )

        self.norm1 = Norm(normalization, num_features=in_channels)
        self.act1 = Activation(activation)
        self.att = Attention(attention, in_channels=in_channels)

        self.ch_pool = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            padding=0,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        norm_channels = in_channels if preactivate else self.out_channels
        self.norm2 = Norm(normalization, num_features=norm_channels)
        self.act2 = Activation(activation)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        x = self.depth_conv(x)
        x = self.norm1(x)
        x = self.act1(x)

        if not self.preattend:
            x = self.att(x)

        # pointwise channel pool
        x = self.ch_pool(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass ith pre-activation."""
        if self.preattend:
            x = self.att(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.depth_conv(x)

        if not self.preattend:
            x = self.att(x)

        # pointwise channel pool
        x = self.norm2(x)
        x = self.act2(x)
        x = self.ch_pool(x)

        return x


class InvertedBottleneckConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size: int = 3,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Mobile inverted bottleneck conv parent-class.

        MobileNetV2: Inverted Residuals and Linear Bottlenecks:
            - https://arxiv.org/abs/1801.04381

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            expand_ratio : float, default=4.0
                The ratio of channel expansion in the bottleneck.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
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
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()

        self.conv_choice = convolution
        self.out_channels = out_channels
        self.preattend = preattend
        self.preactivate = preactivate

        mid_channels = make_divisible(in_channels * expand_ratio)

        self.ch_pool = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            padding=0,
            out_channels=mid_channels,
            kernel_size=1,
            bias=False,
        )

        norm_channels = in_channels if preactivate else mid_channels
        self.norm1 = Norm(normalization, num_features=norm_channels)
        self.act1 = Activation(activation)

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.depth_conv = Conv(
            name=self.conv_choice,
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            groups=mid_channels,
            padding=padding,
            bias=False,
        )
        self.norm2 = Norm(normalization, num_features=mid_channels)
        self.act2 = Activation(activation)

        # set attention
        att_channels = in_channels if preattend else mid_channels
        self.att = Attention(attention, in_channels=att_channels, squeeze_ratio=0.04)

        self.proj_conv = Conv(
            name=self.conv_choice,
            in_channels=mid_channels,
            bias=False,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
        )

        norm_channels = mid_channels if preactivate else self.out_channels
        self.norm3 = Norm(normalization, num_features=norm_channels)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        # pointwise channel pooling conv
        x = self.ch_pool(x)
        x = self.norm1(x)
        x = self.act1(x)

        # depthwise conv
        x = self.depth_conv(x)
        x = self.norm2(x)
        x = self.act2(x)

        if not self.preattend:
            x = self.att(x)

        # Pointwise linear projection
        x = self.proj_conv(x)
        x = self.norm3(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        if self.preattend:
            x = self.att(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.ch_pool(x)

        # depthwise conv
        x = self.norm2(x)
        x = self.act2(x)
        x = self.depth_conv(x)

        if not self.preattend:
            x = self.att(x)

        # Pointwise linear projection
        x = self.norm3(x)
        x = self.proj_conv(x)

        return x


class FusedMobileInvertedConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        same_padding: bool = True,
        convolution: str = "conv",
        preactivate: bool = False,
        normalization: str = "bn",
        activation: str = "relu",
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Fused mobile inverted conv block parent-class.

        Efficientnet-edgetpu: Creating accelerator-optimized neural networks with automl
            - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

        EfficientNetV2: Smaller Models and Faster Training
            - https://arxiv.org/abs/2104.00298

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            expand_ratio : float, default=4.0
                The ratio of channel expansion in the bottleneck.
            kernel_size : int, default=3
                The size of the convolution kernel.
            stride : int, default=1
                The stride size.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
            activation : str, default="relu"
                Activation method.
                One of: "mish", "swish", "relu", "relu6", "rrelu", "selu",
                "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
            convolution : str, default="conv"
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            preactivate : bool, default=False
                If True, normalization will be applied before convolution.
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()
        assert stride in (1, 2)

        self.conv_choice = convolution
        self.out_channels = out_channels
        self.preattend = preattend
        self.preactivate = preactivate

        mid_channels = make_divisible(in_channels * expand_ratio)

        # set padding. Works if dilation or stride are not adjusted
        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv1 = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            padding=padding,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            bias=False,
            stride=stride,
        )

        norm_channels = in_channels if preactivate else mid_channels
        self.norm1 = Norm(normalization, num_features=norm_channels)

        att_channels = in_channels if preattend else mid_channels
        self.att = Attention(attention, in_channels=att_channels, squeeze_ratio=0.04)

        self.proj_conv = Conv(
            name=self.conv_choice,
            in_channels=mid_channels,
            bias=False,
            out_channels=self.out_channels,
            kernel_size=1,
            padding=0,
        )

        norm_channels = mid_channels if preactivate else self.out_channels
        self.norm2 = Norm(normalization, num_features=norm_channels)
        self.act = Activation(activation)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        # pointwise channel pooling conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        if not self.preattend:
            x = self.att(x)

        # Pointwise linear projection
        x = self.proj_conv(x)
        x = self.norm2(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        if self.preattend:
            x = self.att(x)

        # pointwise channel pooling conv
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        if not self.preattend:
            x = self.att(x)

        # Pointwise linear projection
        x = self.norm2(x)
        x = self.proj_conv(x)

        return x


class HoverNetDenseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float = 4.0,
        groups: int = 4,
        same_padding: bool = True,
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        kernel_size: int = 3,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Dense block of the HoVer-Net.

        HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

        Parameters
        ----------
            in_channels : int
                Number of input channels.
            out_channels : int
                Number of output channels.
            squeeze_ratio : float, default=4.0
                The ratio of channel expansion in the bottleneck.
            groups : int, default=1
                Number of groups the kernels are divided into. If `groups == 1`
                normal convolution is applied. If `groups = in_channels`
                depthwise convolution is applied.
            same_padding : bool, default=True
                if True, performs same-covolution.
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
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
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
        """
        super().__init__()

        self.conv_choice = convolution
        self.out_channels = out_channels
        self.preattend = preattend
        self.preactivate = preactivate

        mid_channels = make_divisible(in_channels / squeeze_ratio)
        self.ch_pool = Conv(
            name=self.conv_choice,
            in_channels=in_channels,
            padding=0,
            out_channels=mid_channels,
            kernel_size=1,
            bias=False,
        )

        norm_channels = in_channels if preactivate else mid_channels
        self.norm1 = Norm(normalization, num_features=norm_channels)
        self.act1 = Activation(activation)

        padding = (kernel_size - 1) // 2 if same_padding else 0
        self.conv = Conv(
            name=self.conv_choice,
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=padding,
            bias=False,
        )
        norm_channels = mid_channels if preactivate else out_channels
        self.norm2 = Norm(normalization, num_features=norm_channels)
        self.act2 = Activation(activation)

        # set attention
        att_channels = in_channels if preattend else mid_channels
        self.att = Attention(attention, in_channels=att_channels, squeeze_ratio=0.04)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation."""
        if self.preattend:
            x = self.att(x)

        x = self.ch_pool(x)
        x = self.norm1(x)
        x = self.act1(x)

        if not self.preattend:
            x = self.att(x)

        # pointwise channel pool
        x = self.conv(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x

    def forward_features_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.preattend:
            x = self.att(x)

        x = self.norm1(x)
        x = self.act1(x)
        x = self.ch_pool(x)

        if not self.preattend:
            x = self.att(x)

        # pointwise channel pool
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv(x)

        return x
