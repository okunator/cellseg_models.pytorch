from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .decoder_stage import DecoderStage

__all__ = ["Decoder"]


class Decoder(nn.ModuleDict):
    def __init__(
        self,
        enc_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        style_channels: int = None,
        n_layers: Tuple[int, ...] = (1, 1, 1, 1, 1),
        n_blocks: Tuple[Tuple[int, ...], ...] = ((2,), (2,), (2,), (2,), (2,)),
        long_skip: str = "unet",
        n_transformers: Tuple[int, ...] = None,
        n_transformer_blocks: Tuple[Tuple[int], ...] = ((1,), (1,), (1,), (1,), (1,)),
        stage_params: Optional[Tuple[Dict, ...]] = None,
        **kwargs,
    ) -> None:
        """Build a generic U-net-like decoder.

        Parameters
        ----------
            enc_channels : Tuple[int, ...]
                Number of channels at each encoder layer.
            out_channels : Tuple[int, ...], default=(256, 128, 64, 32, 16)
                Number of channels at each decoder layer output.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
            n_layers : Tuple[int, ...], default=(1, 1, 1, 1, 1)
                The number of conv layers inside each of the decoder stages.
            n_blocks : Tuple[Tuple[int, ...], ...] = ((2, ),(2, ),(2, ),(2, ),(2, ))
                The number of blocks inside each conv-layer at each decoder stage.
            long_skip : str, default="unet"
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            n_transformers : Tuple[int, ...], optional, default=None
                The number of transformer layers inside each of the decoder stages.
            n_transformer_blocks : Tuple[Tuple[int]] = ((1, ),(1, ),(1, ),(1, ),(1, ))
                The number of transformer blocks inside each transformer-layer at each
                decoder stage.
            stage_params : Optional[Tuple[Dict, ...]], default=None
                The keyword args for each of the distinct decoder stages. Incudes the
                parameters for the long skip connections, convolutional layers of the
                decoder and transformer layers itself. See the `DecoderStage`
                documentation for more info.

        Raises
        ------
            ValueError:
                If there is a mismatch between encoder and decoder channel lengths.
        """
        super().__init__()
        self.long_skip = long_skip

        if not len(out_channels) == len(enc_channels):
            raise ValueError(
                "The number of encoder channels need to match the number of "
                f"decoder channels. Got {len(out_channels)} decoder channels "
                f"and {len(enc_channels)} encoder channels."
            )

        out_channels = [enc_channels[0]] + list(out_channels)
        skip_channels = enc_channels[1:]

        # scaling factor assumed to be 2 for the spatial dims and the input
        # has to be divisible by 32. 256 used here just for convenience.
        depth = len(out_channels)
        out_dims = [256 // 2**i for i in range(depth)][::-1]

        # Build decoder
        for i in range(depth - 1):
            # number of conv layers
            n_conv_layers = None
            if n_layers is not None:
                n_conv_layers = n_layers[i]

            # number of conv blocks inside each layer
            n_conv_blocks = None
            if n_blocks is not None:
                n_conv_blocks = n_blocks[i]

            # number of transformer layers
            n_tr_layers = None
            if n_transformers is not None:
                n_tr_layers = n_transformers[i]

            # number of transformer blocks inside transformer layers
            n_tr_blocks = None
            if n_transformer_blocks is not None:
                n_tr_blocks = n_transformer_blocks[i]

            decoder_block = DecoderStage(
                stage_ix=i,
                dec_channels=tuple(out_channels),
                dec_dims=tuple(out_dims),
                skip_channels=skip_channels,
                style_channels=style_channels,
                long_skip=long_skip,
                n_layers=n_conv_layers,
                n_blocks=n_conv_blocks,
                n_transformers=n_tr_layers,
                n_transformer_blocks=n_tr_blocks,
                **stage_params[i] if stage_params is not None else {"k": None},
            )
            self.add_module(f"decoder_stage{i + 1}", decoder_block)

        self.out_channels = decoder_block.out_channels

    def forward_features(
        self, features: Tuple[torch.Tensor], style: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """Forward pass of the decoder. Returns all the decoder stage feats."""
        head = features[0]
        skips = features[1:]
        extra_skips = [head] if self.long_skip == "unet3p" else []
        ret_feats = []

        x = head
        for decoder_stage in self.values():
            x, extra = decoder_stage(
                x, skips=skips, extra_skips=extra_skips, style=style
            )

            if self.long_skip == "unetpp":
                extra_skips = extra
            elif self.long_skip == "unet3p":
                extra_skips.append(x)

            ret_feats.append(x)

        return ret_feats

    def forward(
        self, *features: Tuple[torch.Tensor], style: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        dec_feats = self.forward_features(features, style)

        return dec_feats
