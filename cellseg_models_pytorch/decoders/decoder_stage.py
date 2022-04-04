from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..modules.base_modules import Up
from ..modules.conv_layer import ConvLayer
from .long_skips import LongSkip

__all__ = ["DecoderStage"]


class DecoderStage(nn.Module):
    def __init__(
        self,
        stage_ix: int,
        dec_channels: Tuple[int, ...],
        dec_dims: Tuple[int, ...],
        skip_channels: Tuple[int, ...],
        n_layers: int = 1,
        n_blocks: Tuple[int, ...] = (2,),
        short_skips: Tuple[str, ...] = ("residual",),
        block_types: Tuple[Tuple[str, ...], ...] = (("basic", "basic"),),
        normalizations: Tuple[Tuple[str, ...], ...] = (("bn", "bn"),),
        activations: Tuple[Tuple[str, ...], ...] = (("relu", "relu"),),
        convolutions: Tuple[Tuple[str, ...], ...] = (("conv", "conv"),),
        attentions: Tuple[Tuple[str, ...], ...] = ((None, "se"),),
        preactivates: Tuple[Tuple[bool, ...], ...] = ((False, False),),
        preattends: Tuple[Tuple[bool, ...], ...] = ((False, False),),
        upsampling: str = "fixed-unpool",
        long_skip: str = "unet",
        merge_policy: str = "sum",
        skip_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Build a decoder stage.

        Operations in each decoder stage:
        1. Upsample
        2. Long skip from encoder to decoder if not the last stage.
        3. Conv block

        Parameters
        ----------
            stage_ix : int
                The index number of the current decoder stage.
            dec_channels : Tuple[int, ...]
                The number of output channels in the decoder output stages. First elem
                is the number of channels in the encoder head or bottleneck.
            dec_dims : Tuple[int, ...], default=None
                Tuple of the heights/widths of each encoder/decoder feature map
                e.g. (8, 16, 32, 64, 128, 256). Feature maps are assumed to be square.
            skip_channels : Tuple[int, ...]:
                List of the number of channels in the encoder skip tensors. Ignored if
                `long_skip` == None.
            n_layers : int, default=1
                The number of conv layers inside one decoder stage.
            n_blocks : int, default=2
                Number of conv-blocks inside one conv layer.
            short_skips : str, default=("residual", )
                The short skip methods used inside the conv layers.
            block_types : Tuple[Tuple[str, ...], ...], default=(("basic", "basic"), )
                The type of the convolution blocks in the conv blocks inside the layers.
            normalizations : Tuple[Tuple[str, ...], ...], default: (("bn", "bn"), )
                Normalization methods used in the conv blocks inside the conv layers.
            activations : Tuple[Tuple[str, ...], ...], default: (("relu", "relu"), )
                Activation methods used inside the conv layers.
            attentions : Tuple[Tuple[str, ...], ...], default: ((None, "se"), )
                Attention methods used inside the conv layers.
            preactivates Tuple[Tuple[bool, ...], ...], default: ((False, False), )
                Boolean flags for the conv layers to use pre-activation.
            preattends Tuple[Tuple[bool, ...], ...], default: ((False, False), )
                Boolean flags for the conv layers to use pre-activation.
            upsampling : str, default="fixed-unpool"
                Name of the upsampling method.
            long_skip : str, default="unet"
                long skip method to be used. One of: "unet", "unetpp", "unet3p", None
            merge_policy : str, default="sum"
                The long skip merge policy. One of: "sum", "cat"
            skip_params : Optional[Dict]
                Extra keyword arguments for the skip-connection module. These depend
                on the skip module. Refer to specific skip modules for more info.

        Raises
        ------
            ValueError:
                If lengths of the tuple arguments are not equal to `n_layers`.
        """
        super().__init__()

        illegal_args = [
            (k, a)
            for k, a in locals().items()
            if isinstance(a, tuple)
            and a not in (skip_channels, dec_channels, dec_dims)
            and len(a) != n_layers
        ]

        if illegal_args:
            raise ValueError(
                f"""
                All the tuple-arg lengths need to be equal to `n_layers`={n_layers}.
                Illegal args: {illegal_args}"""
            )

        self.long_skip = long_skip
        self.stage_ix = stage_ix
        self.in_channels = dec_channels[stage_ix]
        self.out_channels = dec_channels[stage_ix + 1]

        # upsampling method
        self.upsample = Up(upsampling)

        # long skip connection method
        self.skip = LongSkip(
            name=long_skip,
            merge_policy=merge_policy,
            stage_ix=self.stage_ix,
            in_channels=self.in_channels,
            dec_channels=dec_channels,
            skip_channels=skip_channels,
            dec_dims=dec_dims,
            **skip_params if skip_params is not None else {"k": None},
        )

        # Set up n layers of conv blocks
        self.conv_layers = nn.ModuleDict()
        for i in range(n_layers):
            n_in_feats = self.skip.out_channels if i == 0 else self.out_channels
            layer = ConvLayer(
                name=block_types[i],
                short_skip=short_skips[i],
                in_channels=n_in_feats,
                out_channels=self.out_channels,
                n_blocks=n_blocks[i],
                block_types=block_types[i],
                normalizations=normalizations[i],
                activations=activations[i],
                convolutions=convolutions[i],
                preactivates=preactivates[i],
                preattends=preattends[i],
                attentions=attentions[i],
            )
            self.conv_layers[f"conv_layer{i + 1}"] = layer
            in_channels = layer.out_channels

        self.out_channels = in_channels

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
        extra_skips: Tuple[torch.Tensor, ...] = None,
    ) -> Tuple[torch.Tensor, Union[None, Tuple[torch.Tensor, ...]]]:
        """Forward pass of the decoder stage.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Shape (B, C, H, W).
            skips : Tuple[torch.Tensor, ...]
                All of feature maps from consecutive encoder blocks.
                Order is bottom up. Shapes: (B, C, H, W).
            extra_skips : Tuple[torch.Tensor, ...], default=None
                Extra skip connections. Used in unet3+ and unet++.

        Returns
        -------
            Tuple[torch.Tensor, Union[None, Tuple[torch.Tensor, ...]]]:
                Output torch.Tensor and extra skip torch.Tensors. If no extra
                skips are present, returns None as the second return value.
        """
        x = self.upsample(x)

        # long skip
        x = self.skip(x, ix=self.stage_ix, skips=skips, extra_skips=extra_skips)

        # unetpp returns extra skips
        extra_skips = x[1] if self.long_skip == "unetpp" else None
        x = x[0] if self.long_skip == "unetpp" else x

        # conv layer
        for _, conv_layer in self.conv_layers.items():
            x = conv_layer(x)

        return x, extra_skips
