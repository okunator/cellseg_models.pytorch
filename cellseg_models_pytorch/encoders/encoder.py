from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .encoder_upsampler import EncoderUpsampler
from .timm_encoder import TimmEncoder

__all__ = ["Encoder"]


class Encoder(nn.Module):
    def __init__(
        self,
        timm_encoder_name: str,
        timm_encoder_out_indices: Tuple[int, ...],
        pixel_decoder_out_channels: Tuple[int, ...],
        timm_encoder_pretrained: bool = True,
        timm_extra_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Wrap timm encoders to one class.

        Parameters
        ----------
        timm_encoder_name : str
            Name of the encoder. If the name is in `TR_ENCODERS.keys()`, a transformer
            will be used. Otherwise, a timm encoder will be used.
        timm_encoder_out_indices : Tuple[int], optional
            Indices of the output features.
        pixel_decoder_out_channels : Tuple[int], optional
            Number of output channels at each upsampling stage.
        timm_encoder_pretrained : bool, optional, default=False
            If True, load pretrained timm weights, by default False.
        timm_extra_kwargs : Dict[str, Any], optional, default={}
            Key-word arguments for any `timm` based encoder. These arguments are
            used in `timm.create_model(**kwargs)` function call.
        """
        super().__init__()

        # initialize timm encoder
        self.encoder = TimmEncoder(
            timm_encoder_name,
            pretrained=timm_encoder_pretrained,
            out_indices=timm_encoder_out_indices,
            extra_kwargs=timm_extra_kwargs,
        )

        # initialize feature upsampler if encoder is a vision transformer
        feature_info = self.encoder.feature_info
        reductions = [finfo["reduction"] for finfo in feature_info]
        if all(element == reductions[0] for element in reductions):
            self.encoder = EncoderUpsampler(
                backbone=self.encoder,
                out_channels=pixel_decoder_out_channels,
            )
            feature_info = self.encoder.feature_info

        self.out_channels = [f["num_chs"] for f in self.encoder.feature_info][::-1]
        self.feature_info = self.encoder.feature_info[::-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of the encoder and return all the features."""
        output, feats = self.encoder(x)
        return output, feats[::-1]
