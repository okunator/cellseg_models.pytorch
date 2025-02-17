from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .timm_encoder import TimmEncoder

__all__ = ["Encoder"]


class Encoder(nn.Module):
    def __init__(
        self,
        timm_encoder_name: str,
        timm_encoder_out_indices: Tuple[int, ...],
        timm_encoder_pretrained: bool = True,
        timm_extra_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Wrap timm encoders to one class.

        Parameters:
            timm_encoder_name (str):
                Name of the encoder. If the name is in `TR_ENCODERS.keys()`, a transformer
                will be used. Otherwise, a timm encoder will be used.
            timm_encoder_out_indices (Tuple[int, ...]):
                Indices of the output features.
            timm_encoder_pretrained (bool, default=True):
                If True, load pretrained timm weights.
            timm_extra_kwargs (Dict[str, Any], default={}):
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

        self.out_channels = [f["num_chs"] for f in self.encoder.feature_info]
        self.feature_info = self.encoder.feature_info  # bottleneck last element

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of the encoder and return all the features."""
        output, feats = self.encoder(x)
        return output, feats  # bottleneck feature is the last element

    def freeze_encoder(self) -> None:
        """Freeze the parameters of the encoeder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
