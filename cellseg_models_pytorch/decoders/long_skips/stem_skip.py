import torch
import torch.nn as nn

from ...modules import ConvLayer
from .merging import Merge

__all__ = ["StemSkip"]


class StemSkip(nn.Module):
    def __init__(
        self,
        out_channels: int,
        merge_policy: str = "cat",
        in_channels: int = 3,
        n_blocks: int = 2,
        short_skip: str = "residual",
        block_type: str = "basic",
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        kernel_size: int = 3,
        groups: int = 1,
        bias: bool = True,
        preactivate: bool = False,
        attention: str = None,
        preattend: bool = False,
        **kwargs,
    ) -> None:
        """Stem skip connection.

        I.e. High resolution skip connection from the input image to the final decoder
        stage.

        Args:
            out_channels: Number of output channels.
            merge_policy:
                Merge policy to be used for the skip connection. Allowed: "cat", "add",
                "none".
                Defaults to "cat".
            in_channels: Number of input channels.
                Defaults to 3.
            short_skip:
                The name of the short skip method. One of: "residual", "dense", "basic"
                Defaults to "residual".
            block_type:
                The name of the conv-block. One of: "basic". "mbconv", "fmbconv" "dws",
                "bottleneck".
            normalization: Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", "lrn", None
                Defaults to "bn".
            activation: Activation method.
                One of: "mish", "swish", "relu", "relu6", "rrelu", "selu",
                "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
                Defaults to "relu".
            convolution:
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
                Defaults to "conv".
            kernel_size: The size of the convolution kernel.
                Defaults to 3.
            groups: Number of groups the kernels are divided into. If `groups == 1`
                normal convolution is applied. If `groups = in_channels`
                depthwise convolution is applied.
                Defaults to 1.
            bias:
                Include bias term in the convolution block. Only used for `BaasicConv`.
                Defaults to True.
            preactivate: If True, normalization will be applied before convolution.
                Defaults to False.
            attention:
                Attention method. One of: "se", "scse", "gc", "eca", "msca", None
                Defaults to None.
            preattend:
                If True, Attention is applied at the beginning of forward pass.
                Defaults to False.
            **kwargs:
                Additional arguments to be passed to the `ConvLayer` and `Merge`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # stem conv
        self.stem_conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            layer_residual=False,
            short_skip=short_skip,
            style_channels=None,
            block_types=(block_type,) * n_blocks,
            normalizations=(normalization,) * n_blocks,
            activations=(activation,) * n_blocks,
            convolutions=(convolution,) * n_blocks,
            kernel_sizes=(kernel_size,) * n_blocks,
            groups=(groups,) * n_blocks,
            biases=(bias,) * n_blocks,
            preactivates=(preactivate,) * n_blocks,
            attentions=(attention,) * n_blocks,
            preattends=(preattend,) * n_blocks,
            **kwargs,
        )

        # skip merge
        self.skip = Merge(
            name=merge_policy,
            in_channels=self.stem_conv.out_channels,
            skip_channels=(out_channels,),
            **kwargs,
        )

        # out conv to orig `out channels`
        self.out_conv = ConvLayer(
            in_channels=self.skip.out_channels,
            out_channels=out_channels,
            n_blocks=n_blocks,
            layer_residual=False,
            short_skip=short_skip,
            style_channels=None,
            block_types=(block_type,) * n_blocks,
            normalizations=(normalization,) * n_blocks,
            activations=(activation,) * n_blocks,
            convolutions=(convolution,) * n_blocks,
            kernel_sizes=(kernel_size,) * n_blocks,
            groups=(groups,) * n_blocks,
            biases=(bias,) * n_blocks,
            preactivates=(preactivate,) * n_blocks,
            attentions=(attention,) * n_blocks,
            preattends=(preattend,) * n_blocks,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass of the stem skip."""
        stem_feat = self.stem_conv(x)
        dec_feat = self.skip(stem_feat, (dec_feat,))
        dec_feat = self.out_conv(dec_feat)

        return dec_feat
