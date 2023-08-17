from typing import Tuple

import torch
import torch.nn as nn

from ...modules import ConvLayer, Up
from .merging import Merge

__all__ = ["UnetppSkip"]


class UnetppSkip(nn.ModuleDict):
    def __init__(
        self,
        stage_ix: int,
        dec_channels: Tuple[int, ...],
        skip_channels: Tuple[int, ...],
        up_factors: Tuple[int, ...],
        hid_channels: int = 256,
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
        merge_policy: str = "cat",
        **kwargs,
    ) -> None:
        """Unet++-like skip connection block with added flexibility.

        UNet++: A Nested U-Net Architecture for Medical Image Segmentation
            - https://arxiv.org/abs/1807.10165

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
                The upscaling factors for each decoder stage.
            hid_channels : int, default=256
                Number of output channels from the hidden middle blocks of unet++.
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
            groups : int, default=(1, 1)
                Number of groups for the kernels in each convolution blocks.
            biases : bool, default=(True, True)
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
        self.stage_ix = stage_ix
        self.skip_channels = skip_channels

        # hot-fix
        kwargs.pop("in_channels") if "in_channels" in kwargs.keys() else kwargs

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

        if stage_ix < len(skip_channels):
            curr_channels = skip_channels[stage_ix]
            prev_channels = skip_channels[stage_ix - 1]

            mid_channels = []
            fin_channels = [curr_channels]

            for i in range(stage_ix):
                up_scale = Up("fixed-unpool", scale_factor=up_factors[self.stage_ix])
                self.add_module(f"up_scale{i + 1}", up_scale)

                channels = [prev_channels] if i == 0 else mid_channels + [hid_channels]
                merge = Merge(
                    merge_policy,
                    in_channels=curr_channels,
                    skip_channels=tuple(channels),
                    **kwargs,
                )
                self.add_module(f"mid_merge{i + 1}", merge)

                for j in range(n_layers):
                    layer = ConvLayer(
                        in_channels=merge.out_channels,
                        out_channels=hid_channels,
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
                    self.add_module(f"mid_layer{i + 1}", layer)

                # previous mid block channels
                fin_channels.append(layer.out_channels)
                mid_channels.append(layer.out_channels)  # new mid block channels

            final_merge = Merge(
                merge_policy,
                in_channels=dec_channels[stage_ix],
                skip_channels=tuple(fin_channels),
                **kwargs,
            )
            self.add_module("final_merge", final_merge)
        else:
            # place-holder Merge-for the final decoder stage
            self.add_module(
                "final_merge",
                Merge(
                    None,
                    in_channels=dec_channels[stage_ix],
                    out_channels=dec_channels[stage_ix],
                ),
            )

    @property
    def out_channels(self) -> int:
        """Out channels."""
        return self.final_merge.out_channels

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor],
        extra_skips: Tuple[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the Unet++ skip connection.

        Parameters
        ----------
            x : torch.Tensor
                Input tensor. Shape (B, C, H, W).
            skips : Tuple[torch.Tensor, ...]
                All of feature maps from consecutive encoder blocks.
                Order is bottom up. Shapes: (B, C, H, W).
            extra_skips : Tuple[torch.Tensor, ...], default=None
                Extra skip connections. (Previous mid features).

        Returns
        -------
            torch.Tensor:
                Output torch.Tensor. Shape: (B, C, H, W).
        """
        mid_features = []
        fin_features = []
        if self.stage_ix < len(skips):
            current_skip = skips[self.stage_ix]
            prev_skip = skips[self.stage_ix - 1]
            fin_features = [current_skip]

            upscales = [k for k in self.keys() if "up_scale" in k]
            merges = [k for k in self.keys() if "mid_merge" in k]
            layers = [k for k in self.keys() if "mid_layer" in k]
            for i, (up, merge, layer) in enumerate(zip(upscales, merges, layers)):
                upskip = prev_skip if i == 0 else extra_skips[i - 1]
                prev_feat = self[up](upskip)

                mid_feat = self[merge](current_skip, [prev_feat] + mid_features)
                mid_feat = self[layer](mid_feat)
                mid_features.append(mid_feat)

            fin_features = fin_features + mid_features

        x = self.final_merge(x, tuple(fin_features))

        return x, mid_features
