from typing import Dict, Optional, Tuple

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
            n_blocks : Tuple[Tuple[int, ...], ...] = ((2, ), (2, ), (2, ). (2, ), (2, ))
                The number of blocks inside each conv-layer in each decoder stage.
            long_skip : str, default="unet"
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            stage_params : Optional[Tuple[Dict, ...]], default=None
                The keyword args for each of the distinct decoder stages. Incudes the
                parameters for the long skip connections and convolutional layers of the
                decoder itself. See the `DecoderStage` documentation for more info.
        """
        super().__init__()
        self.long_skip = long_skip

        if not len(out_channels) == len(enc_channels):
            raise ValueError(
                f"""The number of encoder channels need to match the number of
                decoder channels. Got {len(out_channels)} decoder channels
                and {len(enc_channels)} encoder channels"""
            )

        out_channels = [enc_channels[0]] + list(out_channels)
        skip_channels = enc_channels[1:]

        # scaling factor assumed to be 2 for the spatial dims and the input
        # has to be divisible by 32. 256 used here just for convenience.
        depth = len(out_channels)
        out_dims = [256 // 2**i for i in range(depth)][::-1]

        # Build decoder
        for i in range(depth - 1):
            decoder_block = DecoderStage(
                stage_ix=i,
                dec_channels=tuple(out_channels),
                dec_dims=tuple(out_dims),
                skip_channels=skip_channels,
                style_channels=style_channels,
                long_skip=long_skip,
                n_layers=n_layers[i],
                n_blocks=n_blocks[i],
                **stage_params[i] if stage_params is not None else {"k": None},
            )
            self.add_module(f"decoder_stage{i + 1}", decoder_block)

        self.out_channels = decoder_block.out_channels

    def forward(
        self, *features: Tuple[torch.Tensor], style: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        head = features[0]
        skips = features[1:]
        extra_skips = [head] if self.long_skip == "unet3p" else []

        x = head
        for _, decoder_stage in enumerate(self.values()):
            x, extra = decoder_stage(
                x, skips=skips, extra_skips=extra_skips, style=style
            )

            if self.long_skip == "unetpp":
                extra_skips = extra
            elif self.long_skip == "unet3p":
                extra_skips.append(x)

        return x
