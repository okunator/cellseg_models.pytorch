from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from .histo_encoder import build_histo_encoder
from .timm_encoder import TimmEncoder
from .unettr_encoder import EncoderUnetTR
from .vit_det_SAM import build_sam_encoder

__all__ = ["Encoder"]


TR_ENCODERS = {
    "histo_encoder_prostate_s": build_histo_encoder,
    "histo_encoder_prostate_m": build_histo_encoder,
    "sam_vit_l": build_sam_encoder,
    "sam_vit_b": build_sam_encoder,
    "sam_vit_h": build_sam_encoder,
}


class Encoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        checkpoint_path: str = None,
        in_channels: int = 3,
        depth: int = 4,
        out_indices: Tuple[int] = None,
        unettr_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Wrap timm conv-based encoders and transformer-based encoders to one class.

        NOTE: Refer to the docstring of the `TimmEncoder` and `EncoderUnetTR` for the
        input key-word arguments (**kwargs).

        Parameters
        ----------
        name : str
            Name of the encoder. If the name is in `TR_ENCODERS.keys()`, a transformer
            will be used. Otherwise, a timm encoder will be used.
        pretrained : bool, optional, default=False
            If True, load imagenet pretrained weights, by default False.
        checkpoint_path : str, optional
            Path to the weights of the encoder. If None, the encoder is initialized
            with imagenet pre-trained weights if `enc_pretrain` argument is set to True
            or with random weights if set to False. Defaults to None.
        in_channels : int, optional
            Number of input channels, by default 3.
        depth : int, optional
            Number of output features, by default 4. Ignored for transformer encoders.
        out_indices : Tuple[int], optional
            Indices of the output features, by default None. If None,
            out_indices is set to range(len(depth)). Overrides the `depth` argument.
        unettr_kwargs : Dict[str, Any]
            Key-word arguments for the transformer encoder. These arguments are used
            only if the encoder is transformer based. Refer to the docstring of the
            `EncoderUnetTR`
        **kwargs : Dict[str, Any]
            Key-word arguments for any `timm` based encoder. These arguments are used
            in `timm.create_model(**kwargs)` function call.
        """
        super().__init__()

        if name not in TR_ENCODERS.keys():
            self.encoder = TimmEncoder(
                name,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                in_channels=in_channels,
                depth=depth,
                out_indices=out_indices,
                **kwargs,
            )
        else:
            self.encoder = EncoderUnetTR(
                backbone=TR_ENCODERS[name](
                    name,
                    pretrained=pretrained,
                    checkpoint_path=checkpoint_path,
                ),
                **unettr_kwargs if unettr_kwargs is not None else {},
            )

        self.out_channels = self.encoder.out_channels
        self.feature_info = self.encoder.feature_info

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass of the encoder and return all the features."""
        return self.encoder(x)
