from typing import Tuple

import torch
import torch.nn as nn

from ...modules import ConvLayer, Up
from .merging import Merge

__all__ = ["Unet3pSkip"]


class Unet3pSkip(nn.ModuleDict):
    def __init__(
        self,
        stage_ix: int,
        dec_channels: Tuple[int, ...],
        skip_channels: Tuple[int, ...],
        up_factors: Tuple[int, ...],
        hid_channels: int = 320,
        n_layers: int = 1,
        n_blocks: Tuple[int, ...] = (1,),
        short_skips: Tuple[str, ...] = ("residual",),
        block_types: Tuple[Tuple[str, ...], ...] = (("basic",),),
        activations: Tuple[Tuple[str, ...], ...] = (("relu",),),
        normalizations: Tuple[Tuple[str, ...], ...] = (("bn",),),
        convolutions: Tuple[Tuple[str, ...], ...] = (("conv",),),
        attentions: Tuple[Tuple[str, ...], ...] = ((None,),),
        kernel_sizes: Tuple[int, ...] = ((3,),),
        groups: Tuple[int, ...] = ((1,),),
        biases: Tuple[Tuple[bool, ...], ...] = ((False,),),
        preactivates: Tuple[Tuple[bool, ...], ...] = ((False,),),
        preattends: Tuple[Tuple[bool, ...], ...] = ((False,),),
        use_styles: Tuple[Tuple[bool, ...], ...] = ((False,),),
        expand_ratios: Tuple[float, float] = ((1.0,),),
        merge_policy: str = "sum",
        lite_version: bool = False,
        **kwargs,
    ) -> None:
        """U-net3+-like skip connection block with added flexibility.

        UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation
            - https://arxiv.org/abs/2004.08790

        Sets up a conv block for the upsampled feature map from the
        previous decoder stage and dynamically sets up the conv blocks
        for the outputs of encoder stages and previous decoder stages.
        The number of these conv blocks depend on the decoder stage ix.

        a lite version can be used where the decoder-to-decoder skips
        are skipped.

        Parameters
        ----------
            stage_ix : int
                The index number of the current decoder stage.
            dec_channels : Tuple[int, ...]
                The number of output channels in the decoder output stages. First elem
                is the number of channels in the encoder head or bottleneck.
            skip_channels : Tuple[int, ...]
                List of the number of channels in the encoder skip tensors.
            up_factors : Tuple[int, ...]
                The upsample factors for the decoder stages.
            hid_channels : int
                Number of output channels from this module.
            n_layers : int, default=1
                The number of conv layers inside one skip stage.
            n_blocks : Tuple[int, ...], default=(1, )
                Number of conv-blocks used at each layer of the skip connection.
            short_skips : str, default=("residual", )
                The short skip methods used inside the conv layers.
            block_types : Tuple[Tuple[str, ...], ...], default=(("basic",), )
                The type of the conv blocks in the conv blocks inside the layers.
            normalizations : Tuple[Tuple[str, ...], ...], default: (("bn",), )
                Normalization methods used in the conv blocks inside the conv layers.
            convolutions : Tuple[str, ...], default=(("conv",),)
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            activations : Tuple[Tuple[str, ...], ...], default: (("relu",), )
                Activation methods used inside the conv layers.
            kernel_sizes : Tuple[int, ...], default=(3, 3)
                The size of the convolution kernels in each conv block.
            groups : int, default=((1,),)
                Number of groups for the kernels in each convolution blocks.
            biases : bool, default=((False,),)
                Include bias terms in the convolution blocks.
            attentions : Tuple[Tuple[str, ...], ...], default: ((None,), )
                Attention methods used inside the conv layers.
            preactivates Tuple[Tuple[bool, ...], ...], default: ((False,), )
                Boolean flags for the conv layers to use pre-activation.
            preattends Tuple[Tuple[bool, ...], ...], default: ((False,), )
                Boolean flags for the conv layers to use pre-activation.
            use_styles : Tuple[Tuple[bool, ...], ...], default=((False,), )
                Boolean flags for the conv layers to add style vectors at each block.
            expand_ratios : Tuple[float, float], default=((1.0, ),)
                Expand ratios for the conv blocks.
            upsampling : str, default="fixed-unpool"
                Name of the upsampling method.
            merge_policy : str, default="sum"
                The long skip merge policy. One of: "sum", "cat"
            lite_version : bool, default=False
                If True, the dense decoder-to-decoder skips are not utilized at all.
                Reduces the model params quite a lot and computational cost.
        """
        super().__init__()

        illegal_args = [
            (k, a)
            for k, a in locals().items()
            if isinstance(a, tuple)
            and a not in (skip_channels, dec_channels, up_factors)
            and len(a) != n_layers
        ]

        if illegal_args:
            raise ValueError(
                f"""
                All the tuple-arg lengths need to be equal to `n_layers`={n_layers}.
                Illegal args: {illegal_args}"""
            )

        if dec_channels[stage_ix] <= 4:
            raise ValueError(
                f"Input channels need to be larger than 4. Got {dec_channels[stage_ix]}"
            )

        self.stage_ix = stage_ix
        self.merge_policy = merge_policy
        self.lite_version = lite_version
        self.n_skips = len(skip_channels) + 1
        self.fin_channels = hid_channels
        self.encoder_skip_channels = skip_channels[stage_ix:]

        if stage_ix < len(skip_channels):
            # get downsampling factors for the encoder2decoder connections
            feat_ups = torch.tensor(up_factors[:-1][::-1])
            down_factors = []
            for i, upf in enumerate(feat_ups):
                down_factors.append(int(upf * feat_ups[i + 1 :].prod() / feat_ups[-1]))

            down_factors = down_factors[::-1]

            # get upsampling factors for the decoder2decoder connections
            up_fax = torch.tensor(up_factors).cumprod(0).tolist()[:-1][::-1]

            # decoder skip channels for the decoder2decoder connections
            decoder_skip_channels = dec_channels[:stage_ix]

            conv_channels = self.conv_channels
            out = conv_channels[0]  # out channels for the first conv op
            cat_channels = conv_channels[-1]  # out channels for the rest conv ops

            # conv-layer for the upsampled features
            for i in range(n_layers):
                layer = ConvLayer(
                    in_channels=dec_channels[stage_ix],
                    out_channels=out,
                    n_blocks=n_blocks[i],
                    expand_ratios=expand_ratios[i],
                    short_skip=short_skips[i],
                    block_types=block_types[i],
                    activations=activations[i],
                    normalizations=normalizations[i],
                    convolutions=convolutions[i],
                    attentions=attentions[i],
                    preattends=preattends[i],
                    preactivates=preactivates[i],
                    use_styles=use_styles[i],
                    kernel_sizes=kernel_sizes[i],
                    groups=groups[i],
                    biases=biases[i],
                )
                self.add_module("nonskip_layer", layer)

            # down-sampling blocks and conv-layers for the encoder2decoder connections
            if self.encoder_skip_channels:
                for i, (in_, down_factor) in enumerate(
                    zip(self.encoder_skip_channels, down_factors)
                ):
                    scale_op = self.get_scale_op(down_factor)
                    self.add_module(f"enc2dec_downscale{i + 1}", scale_op)

                    for j in range(n_layers):
                        layer = ConvLayer(
                            in_channels=in_,
                            out_channels=cat_channels if merge_policy == "cat" else out,
                            n_blocks=n_blocks[j],
                            expand_ratios=expand_ratios[j],
                            short_skip=short_skips[j],
                            block_types=block_types[j],
                            activations=activations[j],
                            normalizations=normalizations[j],
                            convolutions=convolutions[j],
                            attentions=attentions[j],
                            preattends=preattends[j],
                            preactivates=preactivates[j],
                            use_styles=use_styles[j],
                            kernel_sizes=kernel_sizes[j],
                            groups=groups[j],
                            biases=biases[j],
                        )
                        self.add_module(f"enc2dec_layer{i + 1}", layer)

            # up-sampling blocks and conv-layers for the decoder2decoder connections
            up_fax = up_fax[::-1][: len(decoder_skip_channels)][::-1]  # don't ask
            if self.encoder_skip_channels and not lite_version:
                for i, in_ in enumerate(decoder_skip_channels):
                    # works only for consecutively 2x upsampled feature maps so will
                    # fail with the transformer decoders
                    scale_op = self.get_scale_op(1 / (up_fax[i] * 2))
                    self.add_module(f"dec2dec_upscale{i + 1}", scale_op)

                    for j in range(n_layers):
                        layer = ConvLayer(
                            in_channels=in_,
                            out_channels=cat_channels if merge_policy == "cat" else out,
                            n_blocks=n_blocks[j],
                            expand_ratios=expand_ratios[j],
                            short_skip=short_skips[j],
                            block_types=block_types[j],
                            activations=activations[j],
                            normalizations=normalizations[j],
                            convolutions=convolutions[j],
                            attentions=attentions[j],
                            preattends=preattends[j],
                            preactivates=preactivates[j],
                            use_styles=use_styles[j],
                            kernel_sizes=kernel_sizes[j],
                            groups=groups[j],
                            biases=biases[j],
                        )
                        self.add_module(f"dec2dec_layer{i + 1}", layer)

            # Final merge block
            merge = Merge(
                name=merge_policy,  # if stage_ix < self.n_skips - 1 else None,
                in_channels=self.nonskip_layer.out_channels,
                skip_channels=conv_channels,
            )
            self.add_module("merge_block", merge)
        else:
            # place-holder Merge-for the final decoder stage
            self.add_module(
                "merge_block",
                Merge(
                    None,
                    in_channels=dec_channels[stage_ix],
                    out_channels=dec_channels[stage_ix],
                ),
            )

    @property
    def out_channels(self) -> int:
        """Out channels."""
        if self.stage_ix < self.n_skips - 1:
            out_channels = self.fin_channels
        else:
            out_channels = self.merge_block.out_channels

        return out_channels

    @property
    def conv_channels(self) -> Tuple[int, ...]:
        """Convolution channel arithmetic.

        Returns
        -------
            Tuple[int, ...]:
                The number of channels for all the skip conv blocks.
        """
        # cat_channels = None
        out_channels = self.fin_channels
        divider = self.n_skips
        if self.lite_version:
            divider = len(self.encoder_skip_channels) + 1

        conv_channels = [self.fin_channels] * (divider - 1)

        if self.merge_policy == "cat":
            # divide the number of out channels evenly for each conv block
            cat_channels, reminder = divmod(self.fin_channels, divider)

            if self.encoder_skip_channels:
                out_channels = cat_channels + reminder

            conv_channels = [out_channels] + [cat_channels] * (divider - 1)

        return conv_channels

    @staticmethod
    # def get_scale_op(in_size: int, target_size: int) -> nn.Module:
    def get_scale_op(scale_factor: int) -> nn.Module:
        """Get the up/down scaling operation for the feature maps."""
        # scale_factor = in_size / target_size

        if scale_factor > 1:
            scale_op = nn.MaxPool2d(kernel_size=int(scale_factor), ceil_mode=True)
        elif scale_factor < 1:
            scale_op = Up("fixed-unpool", scale_factor=int(1 / scale_factor))
        else:
            scale_op = nn.Identity()

        return scale_op

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor],
        extra_skips: Tuple[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the Unet3+ skip connection.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Shape (B, C, H, W).
            skips : Tuple[torch.Tensor, ...]
                All of feature maps from consecutive encoder blocks.
                Order is bottom up. Shapes: (B, C, H, W).
            extra_skips : Tuple[torch.Tensor, ...], default=None
                Extra skip connections. (Decoder2decoder connections).

        Returns
        -------
            torch.Tensor:
                Output torch.Tensor. Shape: (B, C, H, W).
        """
        # init containers for feature maps
        encoder_features = []
        decoder_features = []

        if self.stage_ix < len(skips):
            skips = skips[self.stage_ix :]
            # forward pass for the non-skip layer
            x = self.nonskip_layer(x)

            # forward passes for the encoder2decoder connections
            enc2dec_scales = [k for k in self.keys() if "enc2dec_downscale" in k]
            enc2dec_layers = [k for k in self.keys() if "enc2dec_layer" in k]
            for i, (scale, layer) in enumerate(zip(enc2dec_scales, enc2dec_layers)):
                encoder_feat = self[scale](skips[i])
                encoder_feat = self[layer](encoder_feat)
                encoder_features.append(encoder_feat)

            # forward passes for the decoder2decoder connections
            dec2dec_scales = [k for k in self.keys() if "dec2dec_upscale" in k]
            dec2dec_layers = [k for k in self.keys() if "dec2dec_layer" in k]
            for i, (scale, layer) in enumerate(zip(dec2dec_scales, dec2dec_layers)):
                decoder_feat = self[scale](extra_skips[i])
                decoder_feat = self[layer](decoder_feat)
                decoder_features.append(decoder_feat)

        # Merge all the feature maps
        skip_features = encoder_features + decoder_features
        print(x.shape, [f.shape for f in skip_features])
        x = self.merge_block(x, skip_features)

        return x
