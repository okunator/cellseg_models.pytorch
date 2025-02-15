from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from ..modules import ChannelPool, ConvLayer, Transformer2D, Up
from .long_skips import LongSkip

__all__ = ["UnetDecoderStage"]


class UnetDecoderStage(nn.Module):
    def __init__(
        self,
        stage_ix: int,
        dec_channels: Tuple[int, ...],
        up_factors: Tuple[int, ...],
        skip_channels: Tuple[int, ...],
        long_skip: str = "unet",
        merge_policy: str = "sum",
        skip_params: Dict[str, Any] = None,
        upsampling: str = "fixed-unpool",
        n_conv_layers: int = 1,
        style_channels: int = None,
        layer_residual: bool = False,
        n_conv_blocks: Tuple[int, ...] = (2,),
        short_skips: Tuple[str, ...] = ("residual",),
        expand_ratios: Tuple[float, float] = ((1.0, 1.0),),
        block_types: Tuple[Tuple[str, ...], ...] = (("basic", "basic"),),
        normalizations: Tuple[Tuple[str, ...], ...] = (("bn", "bn"),),
        activations: Tuple[Tuple[str, ...], ...] = (("relu", "relu"),),
        convolutions: Tuple[Tuple[str, ...], ...] = (("conv", "conv"),),
        attentions: Tuple[Tuple[str, ...], ...] = ((None, "se"),),
        preactivates: Tuple[Tuple[bool, ...], ...] = ((False, False),),
        preattends: Tuple[Tuple[bool, ...], ...] = ((False, False),),
        use_styles: Tuple[Tuple[bool, ...], ...] = ((False, False),),
        kernel_sizes: Tuple[Tuple[int, ...]] = ((3, 3),),
        groups: Tuple[Tuple[int, ...]] = ((1, 1),),
        biases: Tuple[Tuple[bool, ...]] = ((False, False),),
        n_transformers: int = None,
        n_transformer_blocks: Tuple[int, ...] = (1,),
        transformer_blocks: Tuple[Tuple[str, ...], ...] = (("exact",),),
        transformer_computations: Tuple[Tuple[str, ...], ...] = (("basic",),),
        transformer_biases: Tuple[Tuple[bool, ...], ...] = ((False,),),
        transformer_dropouts: Tuple[Tuple[float, ...], ...] = ((0.0,),),
        transformer_layer_scales: Tuple[Tuple[bool, ...], ...] = ((False,),),
        transformer_params: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Build a decoder stage.

        Operations in each decoder stage:
        1. Upsample.
        2. Long skip from encoder to decoder if not the last stage.
        3. Conv block (optional, applied by default).
        4. Transformer block (optional, not applied by default).

        Parameters
        ----------
            stage_ix : int
                The index number of the current decoder stage.
            dec_channels : Tuple[int, ...]
                The number of output channels in the decoder output stages. First elem
                is the number of channels in the encoder head or bottleneck.
            up_factors : Tuple[int, ...]
                The upsampling factors for each decoder stage. The tuple-length has to
                match `dec_channels`.
            skip_channels : Tuple[int, ...]
                List of the number of channels in the encoder skip tensors. Ignored if
                `long_skip` == None.
            long_skip : str, default="unet"
                long skip method to be used.
                Allowed: "cross-attn", "unet", "unetpp", "unet3p", "unet3p-lite", None
            merge_policy : str, default="sum"
                The long skip merge policy. One of: "sum", "cat"
            skip_params : Dict[str, Any], default=None
                Extra keyword arguments for the skip-connection module. These depend
                on the skip module. Refer to specific skip modules for more info.
            upsampling : str, default="fixed-unpool"
                Name of the upsampling method. One of: "fixed-unpool", "bilinear",
                "nearest", "bicubic", "conv_transpose"
            n_conv_layers : int, default=1
                The number of conv layers inside one decoder stage.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
                Also, ignored if `n_conv_layers` is None.
            layer_residual : bool, optional, default=False
                Apply a layer level residual short skip at each layer. I.e x + layer(x).
                Ignored if `n_conv_layers` is None.
            n_conv_blocks : Tuple[int, ...], default=(2,)
                Number of conv-blocks inside each conv layer. The tuple-length has to
                match `n_conv_layers`. Ignored if `n_conv_layers` is None.
            short_skips : str, optional,  default=("residual", )
                The short skip methods used inside the conv layers. Ignored if
                `n_conv_layers` is None.
            expand_ratios : Tuple[float, ...], default=((1.0, 1.0),):
                Expansion/Squeeze ratios for the out channels of each conv block.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            block_types : Tuple[Tuple[str, ...], ...], default=(("basic", "basic"), )
                The type of the convolution blocks in the conv blocks inside the layers.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            normalizations : Tuple[Tuple[str, ...], ...], default: (("bn", "bn"), )
                Normalization methods used in the conv blocks inside the conv layers.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            activations : Tuple[Tuple[str, ...], ...], default: (("relu", "relu"), )
                Activation methods used inside the conv layers.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            attentions : Tuple[Tuple[str, ...], ...], default: ((None, "se"), )
                Channel-attention methods used inside the conv layers.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None. Allowed methods.: "se", "scse", "gc", "eca",
                "msca", None.
            preactivates Tuple[Tuple[bool, ...], ...], default: ((False, False), )
                Boolean flags for the conv layers to use pre-activation.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            preattends Tuple[Tuple[bool, ...], ...], default: ((False, False), )
                Boolean flags for the conv layers to use pre-activation.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            use_styles : Tuple[Tuple[bool, ...], ...], default=((False, False), )
                Boolean flags for the conv layers to add style vectors at each block.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            kernel_sizes : Tuple[int, ...], default=((3, 3),)
                The size of the convolution kernels in each conv block.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            groups : int, default=((1, 1),)
                Number of groups for the kernels in each convolution blocks.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            biases : bool, default=((False, False),)
                Include bias terms in the convolution blocks.
                The tuple-length has to match `n_conv_layers`. Ignored if
                `n_conv_layers` is None.
            n_transformers : int, optional
                Number of self-attention tranformers applied after the conv-layer.
                If this is None, no transformers will be added.
            n_transformer_blocks : int, default=(2, ), optional
                Number of multi-head self attention blocks used in the transformer
                layers. Ignored if `n_transformers` is None.
            transformer_blocks : Tuple[Tuple[str, ...], ...], default=(("basic",),)
                The name of the SelfAttentionBlocks in the TransformerLayer(s).
                Allowed values: "basic", "slice", "flash". Ignored if `n_transformers`
                is None. Length of the tuple has to equal `n_transformer_blocks`.
                Allowed names: ("exact", "linformer").
            transformer_computations : Tuple[Tuple[str, ...],...], default=(("basic",),)
                The way of computing the attention matrices in the SelfAttentionBlocks
                in the TransformerLayer(s). Length of the tuple has to equal
                `n_transformer_blocks`. Allowed styles: "basic". "slice", "flash",
                "memeff", "slice-memeff".
            transformer_biases : Tuple[Tuple[bool, ...], ...], default=((False,),)
                Flags, whether to use biases in the transformer layers. Ignored if
                `n_transformers` is None.
            transformer_dropoouts : Tuple[Tuple[float, ...], ...], default=((0.0,),)
                Dropout probabilities in the transformer layers. Ignored if
                `n_transformers` is None.
            transformer_layer_scales : Tuple[Tuple[bool, ...], ...], default=((False,),)
                Flags, whether to use layer scales in the transformer layers. Ignored if
                `n_transformers` is None.
            transformer_params : List[Dict[str, Any]]
                Extra keyword arguments for the transformer layers. Refer to
                `Transformer2D` module for more info. Ignored if `n_transformers`
                is None.

        Raises
        ------
            ValueError:
                If lengths of the conv layer tuple args are not equal to `n_conv_layers`
                If lengths of the transformer layer tuple args are not equal to
                `n_transformers`.
        """
        super().__init__()

        self.n_conv_layers = n_conv_layers
        self.n_transformers = n_transformers
        self.long_skip = long_skip
        self.stage_ix = stage_ix
        self.in_channels = dec_channels[stage_ix]
        self.out_channels = dec_channels[stage_ix + 1]

        # upsampling method
        self.upsample = Up(
            upsampling,
            scale_factor=up_factors[stage_ix],
            in_channels=self.in_channels,  # needed for transconv (otherwise ignored)
            out_channels=self.out_channels,  # needed for transconv (otherwise ignored)
        )

        # long skip connection method
        # if upsampling is conv_transpose, the skip_in_chans is the out_channels
        # of convtranspose module
        skip_in_chans = self.in_channels
        if upsampling == "conv_transpose":
            skip_in_chans = self.out_channels

        self.skip = LongSkip(
            name=long_skip,
            merge_policy=merge_policy,
            stage_ix=self.stage_ix,
            in_channels=skip_in_chans,
            dec_channels=dec_channels,
            skip_channels=skip_channels,
            up_factors=up_factors,
            **skip_params if skip_params is not None else {"k": None},
        )

        # Set up n layers of conv blocks
        layer = None  # placeholder
        if n_conv_layers is not None:
            # check that the conv-layer tuple-args are not illegal.
            self._check_tuple_args(
                "conv-layer related",
                "n_conv_layers",
                n_conv_layers,
                all_args=locals(),
                skip_args=(
                    skip_channels,
                    dec_channels,
                    up_factors,
                    transformer_blocks,
                    transformer_computations,
                    n_transformer_blocks,
                    transformer_biases,
                    transformer_dropouts,
                    transformer_layer_scales,
                ),
            )

            # set up the conv-layers.
            self.conv_layers = nn.ModuleDict()
            for i in range(n_conv_layers):
                n_in_feats = self.skip.out_channels if i == 0 else layer.out_channels
                layer = ConvLayer(
                    in_channels=n_in_feats,
                    out_channels=self.out_channels,
                    n_blocks=n_conv_blocks[i],
                    layer_residual=layer_residual,
                    style_channels=style_channels,
                    short_skip=short_skips[i],
                    expand_ratios=expand_ratios[i],
                    block_types=block_types[i],
                    normalizations=normalizations[i],
                    activations=activations[i],
                    convolutions=convolutions[i],
                    preactivates=preactivates[i],
                    preattends=preattends[i],
                    use_styles=use_styles[i],
                    attentions=attentions[i],
                    kernel_sizes=kernel_sizes[i],
                    groups=groups[i],
                    biases=biases[i],
                    **kwargs,
                )
                self.conv_layers[f"conv_layer{i + 1}"] = layer

            self.out_channels = layer.out_channels

        # set in_channels for final operations
        in_channels = (
            self.skip.out_channels if n_conv_layers is None else self.out_channels
        )

        if n_transformers is not None:
            # check that the transformer-layer tuple args are not illegal.
            self._check_tuple_args(
                "transformer related",
                "n_transformers",
                n_transformers,
                all_args=locals(),
                skip_args=(
                    skip_channels,
                    dec_channels,
                    up_factors,
                    n_conv_blocks,
                    short_skips,
                    expand_ratios,
                    block_types,
                    normalizations,
                    activations,
                    convolutions,
                    attentions,
                    preactivates,
                    preattends,
                    use_styles,
                    kernel_sizes,
                    groups,
                    biases,
                ),
            )

            # set up the transformer layers
            self.transformers = nn.ModuleDict()
            for i in range(n_transformers):
                tr = Transformer2D(
                    in_channels=in_channels,
                    n_blocks=n_transformer_blocks[i],
                    block_types=transformer_blocks[i],
                    computation_types=transformer_computations[i],
                    biases=transformer_biases[i],
                    dropouts=transformer_dropouts[i],
                    layer_scales=transformer_layer_scales[i],
                    **transformer_params
                    if transformer_params is not None
                    else {"k": None},
                )
                self.transformers[f"tr_layer_{i + 1}"] = tr

        # add a channel pooling layer at the end if no conv-layers are set up
        if n_conv_layers is None:
            self.ch_pool = ChannelPool(
                in_channels=in_channels,
                out_channels=self.out_channels,
                normalization=normalizations[0][0],
                convolution=convolutions[0][0],
            )

    def _check_tuple_args(
        self,
        case: str,
        var: str,
        n: int,
        all_args: Dict[str, Any],
        skip_args: Tuple[Any, ...],
    ) -> None:
        """Check for illegal tuple-args."""
        illegal_args = [
            (k, a)
            for k, a in all_args.items()
            if isinstance(a, tuple) and a not in skip_args and len(a) != n
        ]

        if illegal_args:
            raise ValueError(
                f"All {case} tuple-arg lengths need to be equal to `{var}`={n}. "
                f"Illegal args: {illegal_args}"
            )

    def forward(
        self,
        x: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
        extra_skips: Tuple[torch.Tensor, ...] = None,
        style: torch.Tensor = None,
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
            style : torch.Tensor
                Style vector. Shape (B, C).

        Returns
        -------
            Tuple[torch.Tensor, Union[None, Tuple[torch.Tensor, ...]]]:
                Output torch.Tensor and extra skip torch.Tensors. If no extra
                skips are present, returns None as the second return value.
        """
        x = self.upsample(x)  # (B, in_channels, H, W)

        # long skip (B, in_channels(+skip_channels), H, W)
        x = self.skip(x, ix=self.stage_ix, skips=skips, extra_skips=extra_skips)

        # unetpp returns extra skips
        extra_skips = x[1] if self.long_skip == "unetpp" else None
        x = x[0] if self.long_skip == "unetpp" else x

        # conv layers
        if self.n_conv_layers is not None:
            for conv_layer in self.conv_layers.values():
                x = conv_layer(x, style)  # (B, out_channels, H, W)

        # transformer layers
        if self.n_transformers is not None:
            for transformer in self.transformers.values():
                x = transformer(x)  # (B, long_skip_channels/out_channels, H, W)

        # channel pool if conv-layers are skipped.
        if self.n_conv_layers is None:
            x = self.ch_pool(x)  # (B, out_channels, H, W)

        return x, extra_skips
