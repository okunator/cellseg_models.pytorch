from typing import Tuple

import torch
import torch.nn as nn

from ..modules import ConvBlock
from ..modules.upsample import FixedUnpool

__all__ = ["FeatUpSampleBlock", "EncoderUnetTR"]


class FeatUpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_method: str = "conv_transpose",
        short_skip: str = "basic",
        activation: str = "relu",
        normalization: str = "bn",
        conv_block: str = "basic",
        convolution: str = "conv",
        attention: str = None,
        **kwargs,
    ) -> None:
        """Upsample 2D dimensions of a feature.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        up_method : str, default="conv_transpose"
            The upsampling method to be used. One of: "fixed_unpool", "bilinear",
            "nearest", "conv_transpose". Defaults to "conv_transpose".
        short_skip : str, default="basic"
            The short skip method to be used. One of: "basic", "dense", "residual".
            Defaults to "basic".
        activation : str, default="relu"
            The activation function to be used. One of: One of: "mish", "swish", "relu",
            "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh", "sigmoid", "silu",
            "prelu", "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
        normalization : str, default="bn"
            The normalization method to be used. One of: "bn", "bcn", "gn", "in", "ln",
            "lrn", None
        conv_block : str, default="basic"
            The name of the conv-block. One of: "basic". "mbconv", "fmbconv" "dws",
            "bottleneck".
        convolution : str, default="conv"
            The convolution method to be used. One of: "conv", "wsconv", "scaled_wsconv"
        attention : str, optional
            Attention method. One of: "se", "scse", "gc", "eca", "msca", None
        """
        super().__init__()
        allowed = ("fixed_unpool", "bilinear", "nearest", "conv_transpose")
        if up_method not in allowed:
            raise ValueError(f"`up_method` must be in {allowed}, got {up_method}.")

        if up_method == "fixed_unpool":
            self.up = FixedUnpool(scale_factor=2)
        elif up_method == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        elif up_method == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        elif up_method == "conv_transpose":
            if in_channels is None or out_channels is None:
                raise ValueError(
                    "If `up_method` is `conv_transpose`, `in_channels` & `out_channels`"
                    " need to be specified."
                )

            self.up = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )

        self.conv_block = ConvBlock(
            name=conv_block,
            in_channels=out_channels if up_method == "conv_transpose" else in_channels,
            out_channels=out_channels,
            use_style=False,
            short_skip=short_skip,
            activation=activation,
            normalization=normalization,
            attention=attention,
            convolution=convolution,
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsampler stage."""
        x = self.up(x)
        x = self.conv_block(x)
        return x


class EncoderUnetTR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_channels: Tuple[int, ...] = None,
        up_method: str = "conv_transpose",
        short_skip: str = "basic",
        activation: str = "relu",
        normalization: str = "bn",
        convolution: str = "conv",
        conv_block: str = "basic",
        attention: str = None,
        **kwargs,
    ) -> None:
        """Wrap any transformer backbone into the UnetTR-like encoder.

        Parameters
        ----------
        backbone : nn.Module
            A backbone model. The backbone should return a list of 2D features with a
            fixed resolution (`patch_size`).
        out_channels : Tuple[int, ...], optional
            The number of output channels at each upsampling stage. If None, the number
            of output channels is set to the `embed_dim` of the backbone.
        up_method : str, default="conv_transpose"
            The upsampling method to be used. One of: "fixed_unpool", "bilinear",
            "nearest", "conv_transpose". Defaults to "conv_transpose".
        short_skip : str, default="basic"
            The short skip method to be used. One of: "basic", "dense", "residual".
            Defaults to "basic".
        activation : str, default="relu"
            The activation function to be used. One of: One of: "mish", "swish", "relu",
            "relu6", "rrelu", "selu", "celu", "gelu", "glu", "tanh", "sigmoid", "silu",
            "prelu", "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
        normalization : str, default="bn"
            The normalization method to be used. One of: "bn", "bcn", "gn", "in", "ln",
            "lrn", None
        conv_block : str, default="basic"
            The name of the conv-block. One of: "basic". "mbconv", "fmbconv" "dws",
            "bottleneck".
        convolution : str, default="conv"
            The convolution method to be used. One of: "conv", "wsconv", "scaled_wsconv"
        attention : str, optional
            Attention method. One of: "se", "scse", "gc", "eca", "msca", None
        """
        super().__init__()
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.embed_dim = backbone.embed_dim
        self.out_indices = backbone.out_indices

        # check if out_channels is given and if it is of the same length as out_indices
        if out_channels is not None:
            if len(out_channels) != len(self.out_indices):
                raise ValueError(
                    "`out_channels` must be the same length as `backbone.out_indices`"
                    f" Got {out_channels} and {backbone.out_indices} respectively."
                )

        # set the out_channels, if not given, set to the embed_dim
        if out_channels is None:
            self.out_channels = [self.embed_dim] * len(self.out_indices)
        else:
            self.out_channels = out_channels

        self.feature_info = []

        # bottleneck layer
        self.bottleneck = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.out_channels[0],
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        # add timm-like feature info of the bottleneck layer
        self.feature_info.append(
            {
                "num_chs": self.out_channels[0],
                "module": "bottleneck",
                "reduction": self.patch_size // 2,
            }
        )

        self.up_blocks = nn.ModuleDict()
        n_up_blocks = [i for i in range(1, len(self.out_channels))]
        for i, (out_chan, nblocks) in enumerate(
            zip(self.out_channels[1:], n_up_blocks)
        ):
            up_blocks = []
            squeeze_rates = list(range(nblocks))[::-1]
            for j, sr in zip(range(nblocks), squeeze_rates):
                if j == 0:
                    in_channels = self.embed_dim
                else:
                    in_channels = up.out_channels  # noqa

                up = FeatUpSampleBlock(
                    in_channels=in_channels,
                    out_channels=out_chan * (2**sr),
                    up_method=up_method,
                    short_skip=short_skip,
                    activation=activation,
                    normalization=normalization,
                    conv_block=conv_block,
                    convolution=convolution,
                    attention=attention,
                )
                up_blocks.append(up)

            # add feature info
            self.feature_info.append(
                {
                    "num_chs": out_chan,
                    "module": f"up{i + 1}",
                    "reduction": self.patch_size // 2**nblocks,
                }
            )
            self.up_blocks[f"up{i + 1}"] = nn.Sequential(*up_blocks)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the unetTR."""
        feats = self.backbone(x)

        # bottleneck feature
        intermediate_features = []
        up_feat = self.bottleneck(feats[0])
        intermediate_features.append(up_feat)

        # upsampled features
        for i, feat in enumerate(feats[1:]):
            up_feat = self.up_blocks[f"up{i + 1}"](feat)
            intermediate_features.append(up_feat)

        return tuple(intermediate_features)
