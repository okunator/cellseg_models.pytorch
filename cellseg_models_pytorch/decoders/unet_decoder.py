from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .unet_decoder_stage import UnetDecoderStage

__all__ = ["UnetDecoder"]


class UnetDecoder(nn.ModuleDict):
    def __init__(
        self,
        enc_channels: Tuple[int, ...],
        enc_reductions: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        long_skip: Union[None, str, Tuple[str, ...]] = "unet",
        n_conv_layers: Union[None, int, Tuple[int, ...]] = 1,
        n_conv_blocks: Union[int, Tuple[Tuple[int, ...], ...]] = 2,
        n_transformers: Union[None, int, Tuple[int, ...]] = None,
        n_transformer_blocks: Union[int, Tuple[Tuple[int], ...]] = 1,
        stage_params: Tuple[Dict, ...] = None,
        style_channels: int = None,
        **kwargs,
    ) -> None:
        """Build a generic U-net-like decoder.

        I.e stack decoder stages that are composed followingly:

        DecoderStage:
            - UpSample(up_method)
            - LongSkip(long_skip_method)
            - ConvLayer (optional)
                - ConvBlock(conv_block_method)
            - TransformerLayer (optional)
                - TransformerBlock(transformer_block_method)

        Parameters
        ----------
            enc_channels : Tuple[int, ...]
                Number of channels at each encoder layer.
            enc_reductions : Tuple[int, ...]
                The reduction factor from the input image size at each encoder layer.
            out_channels : Tuple[int, ...]
                Number of channels at each decoder layer output.
            long_skip : Union[None, str, Tuple[str, ...]], default="unet"
                long skip method to be used. The argument can be given as a tuple, where
                each value indicates the long-skip method for each stage of the decoder,
                allowing the mixing of long-skip methods in the decoder.
                Allowed: "cross-attn", "unet", "unetpp", "unet3p", "unet3p-lite", None
            n_conv_layers : Union[None, int, Tuple[int, ...]], default=1
                The number of convolution layers inside each of the decoder stages. The
                argument can be given as a tuple, where each value indicates the number
                of conv-layers inside each stage of the decoder allowing the mixing of
                different sized layers inside the stages. If set to None, no conv-layers
                will be included in the decoder.
            n_transformers : Union[None, int, Tuple[int, ...]] , optional
                The number of transformer layers inside each of the decoder stages. The
                argument can be given as a tuple, where each value indicates the number
                of transformer-layers inside each stage of the decoder stages allowing
                the mixing of different sized layers inside the stages. If set to None,
                no transformer layers will be included in the decoder.
            n_conv_blocks : Union[int, Tuple[Tuple[int, ...], ...]], default=2
                The number of blocks inside each conv-layer at each decoder stage. The
                argument can be given as a nested tuple, where each value indicates the
                number of `ConvBlock`s inside a single `ConvLayer` allowing different
                sized blocks inside each conv-layer in the decoder.
            n_transformer_blocks : Union[int, Tuple[Tuple[int], ...]], default=1
                The number of transformer blocks inside each transformer-layer at each
                decoder stage. The argument can be given as a nested tuple, where each
                value indicates the number of `SelfAttention`s inside a single
                `TranformerLayer` allowing different sized transformer blocks inside
                each transformer-layer in the decoder.
            stage_params : Tuple[Dict, ...], default=None
                The keyword args for each of the distinct decoder stages. Incudes the
                parameters for the long skip connections, convolutional layers of the
                decoder and transformer layers itself. See the `DecoderStage`
                documentation for more info.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
                If `n_conv_layers` is None, this is ignored since style vectors are
                applied inside `ConvBlocks`.

        Raises
        ------
            ValueError:
                If there is a mismatch between encoder and decoder channel lengths.
        """
        super().__init__()

        if not len(out_channels) == len(enc_channels):
            raise ValueError(
                "The number of encoder channels need to match the number of "
                f"decoder channels. Got {len(out_channels)} decoder channels "
                f"and {len(enc_channels)} encoder channels."
            )

        out_channels = [enc_channels[0]] + list(out_channels)
        skip_channels = enc_channels[1:]
        self.depth = len(out_channels)

        # convert the reduction factors to upsample factors for the decoder
        enc_reductions = list(enc_reductions)
        enc_reductions.append(1)  # add the final resolution
        up_factors = torch.tensor(enc_reductions)
        up_factors = (up_factors[:-1] / up_factors[1:]).tolist()  # consecutive ratios
        up_factors = [int(f) for f in up_factors]

        # set layer-level tuple-args
        self.long_skips = self._layer_tuple(long_skip)
        n_conv_layers = self._layer_tuple(n_conv_layers)
        n_transformers = self._layer_tuple(n_transformers)

        # set block-level tuple-args
        n_conv_blocks = self._block_tuple(n_conv_blocks, n_conv_layers)
        n_transformer_blocks = self._block_tuple(n_transformer_blocks, n_transformers)

        # Build decoder
        for i in range(self.depth - 1):
            decoder_block = UnetDecoderStage(
                stage_ix=i,
                dec_channels=tuple(out_channels),
                up_factors=tuple(up_factors),
                skip_channels=skip_channels,
                long_skip=self._tup_arg(self.long_skips, i),
                n_conv_layers=self._tup_arg(n_conv_layers, i),
                n_conv_blocks=self._tup_arg(n_conv_blocks, i),
                n_transformers=self._tup_arg(n_transformers, i),
                n_transformer_blocks=self._tup_arg(n_transformer_blocks, i),
                style_channels=style_channels,
                **stage_params[i] if stage_params is not None else {"k": None},
            )
            self.add_module(f"decoder_stage{i + 1}", decoder_block)

        self.out_channels = decoder_block.out_channels

    def _tup_arg(self, tup: Tuple[Any, ...], ix: int) -> Union[None, int, str]:
        """Return None if given tuple-arg is None, else, return the value at ix."""
        ret = None
        if tup is not None:
            ret = tup[ix]
        return ret

    def _layer_tuple(
        self, arg: Union[None, str, int, Tuple[Any, ...]]
    ) -> Union[None, Tuple[Any, ...]]:
        """Return a non-nested tuple or None for layer-related arguments."""
        ret = None
        if isinstance(arg, (list, tuple)):
            ret = tuple(arg)
        elif isinstance(arg, (str, int)):
            ret = tuple([arg] * self.depth)
        elif arg is None:
            ret = ret
        else:
            raise ValueError(
                f"Given arg: {arg} should be None, str, int or a Tuple of ints or strs."
            )

        return ret

    def _block_tuple(
        self,
        arg: Union[int, None, Tuple[Tuple[int, ...], ...]],
        n_layers: Tuple[int, ...],
    ) -> Union[None, Tuple[Tuple[int, ...], ...]]:
        """Return a nested tuple or None for block-related arguments."""
        ret = None
        if isinstance(arg, (list, tuple)):
            if not all([isinstance(a, (tuple, list)) for a in arg]):
                raise ValueError(
                    f"Given arg: {arg} should be a nested sequence. Got: {arg}."
                )
            ret = tuple(arg)
        elif isinstance(arg, int):
            if n_layers is not None:
                ret = tuple([tuple([arg] * i) for i in n_layers])
            else:
                ret = None
        elif arg is None:
            ret = ret
        else:
            raise ValueError(f"Given arg: {arg} should be None, int or a nested tuple.")

        return ret

    def forward_features(
        self, features: Tuple[torch.Tensor], style: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """Forward pass of the decoder. Returns all the decoder stage feats."""
        head = features[0]
        skips = features[1:]
        extra_skips = [head] if self.long_skips[0] == "unet3p" else []
        ret_feats = []

        x = head
        for i, decoder_stage in enumerate(self.values()):
            x, extra = decoder_stage(
                x, skips=skips, extra_skips=extra_skips, style=style
            )

            if self.long_skips[i] == "unetpp":
                extra_skips = extra
            elif self.long_skips[i] == "unet3p":
                extra_skips.append(x)

            ret_feats.append(x)

        return ret_feats

    def forward(
        self, *features: Tuple[torch.Tensor], style: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        dec_feats = self.forward_features(features, style)

        return dec_feats
