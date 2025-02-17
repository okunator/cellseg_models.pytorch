from itertools import chain
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.decoders.long_skips import StemSkip
from cellseg_models_pytorch.decoders.unet_decoder import UnetDecoder
from cellseg_models_pytorch.encoders.encoder_upsampler import EncoderUpsampler
from cellseg_models_pytorch.models.base._initialization import (
    initialize_decoder,
    initialize_head,
)
from cellseg_models_pytorch.models.base._seg_head import SegHead
from cellseg_models_pytorch.modules.misc_modules import StyleReshape

ALLOWED_HEADS = [
    "inst",
    "type",
    "sem",
    "cellpose",
    "omnipose",
    "stardist",
    "hovernet",
    "dist",
    "dcan",
    "dran",
]

__all__ = ["MultiTaskDecoder"]


class MultiTaskDecoder(nn.ModuleDict):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        out_channels: Tuple[int, ...],
        enc_feature_info: Tuple[Dict[str, Any], ...],
        n_layers: Tuple[int, ...],
        n_blocks: Tuple[int, ...],
        stage_kws: Tuple[Dict[str, Any], ...],
        stem_skip_kws: Dict[str, Any] = None,
        long_skip: str = "unet",
        out_size: int = None,
        style_channels: int = None,
        head_excitation_channels: int = None,
    ) -> None:
        """Create a multi-task decoder.

        Parameters:
            decoders (Tuple[str, ...]):
                Tuple of decoder names. E.g. ("decoder1", "decoder2").
            heads (Dict[str, Dict[str, int]]):
                Dict containing the heads for each decoder. The inner dict contains the
                head name and the number of output channels. For example:
                {"decoder1": {"inst": 2, "sem": 5}, "decoder2": {"cellpose": 2}}.
            out_channels (Tuple[int, ...]):
                Tuple of output channels for each decoder stage. The length of the tuple
                should be equal to the number of enc_channels.
            enc_feature_info (Tuple[Dict[str, Any], ...]):
                Tuple of encoder feature info dicts. Basically timm.model.feature_info
            n_layers (Tuple[int, ...]):
                Tuple of number of conv layers in each decoder stage.
            n_blocks (Tuple[int, ...]):
                Tuple of number of conv blocks in each decoder stage.
            stage_kws (Tuple[Dict[str, Any], ...]):
                Tuple of kwargs for each decoder stage. See UnetDecoderStage for info.
            stem_skip_kws (Dict[str, Any], default=None):
                Optional kwargs for the stem skip connection.
            long_skip (str, default="unet"):
                The long skip connection method to be used in the decoder
            out_size (int, default=None):
                The output size of the model. If given, the output will be interpolated to this size.
            style_channels (int, default=None):
                The number of style channels for domain adaptation.
            head_excitation_channels (int, default=None):
                The number of excitation channels for the head. If None, no excitation is
                used. Excitation is a conv block before the head that widens the output
                channels before the head to avoid 'fight over features' (stardist).
        """
        super().__init__()
        self.out_size = out_size
        self._check_head_args(heads, decoders)
        self._check_decoder_args(decoders)
        self._check_depth(
            len(n_blocks),
            {
                "n_layers": n_layers,
                "out_channels": out_channels,
                "enc_feature_info": enc_feature_info,
            },
        )

        # get the reduction factors and out channels of the encoder
        self.enc_feature_info = enc_feature_info[::-1]  # bottleneck first
        enc_reductions = tuple([inf["reduction"] for inf in self.enc_feature_info])
        enc_channels = tuple([inf["num_chs"] for inf in self.enc_feature_info])

        # initialize feature upsampler if encoder is a vision transformer
        self.encoder_upsampler = None
        if all(elem == enc_reductions[0] for elem in enc_reductions):
            self.encoder_upsampler = EncoderUpsampler(
                feature_info=enc_feature_info,
                out_channels=out_channels,
            )
            self.enc_feature_info = self.encoder_upsampler.feature_info  # bottlneck 1st
            enc_reductions = tuple([inf["reduction"] for inf in self.enc_feature_info])
            enc_channels = tuple([inf["num_chs"] for inf in self.enc_feature_info])

        # style
        self.make_style = None
        if style_channels is not None:
            self.make_style = StyleReshape(enc_channels[0], style_channels)

        # set decoders
        for decoder_name in decoders:
            decoder = UnetDecoder(
                enc_channels=enc_channels,
                enc_reductions=enc_reductions,
                out_channels=out_channels,
                style_channels=style_channels,
                long_skip=long_skip,
                n_conv_layers=n_layers,
                n_conv_blocks=n_blocks,
                stage_params=stage_kws,
            )
            self.add_module(f"{decoder_name}_decoder", decoder)

        # optional stem skip
        self.has_stem_skip = stem_skip_kws is not None
        if self.has_stem_skip:
            for decoder_name in decoders:
                stem_skip = StemSkip(out_channels=out_channels[-1], **stem_skip_kws)
                self.add_module(f"{decoder_name}_stem_skip", stem_skip)

        # set heads
        for decoder_name in heads.keys():
            for output_name, n_classes in heads[decoder_name].items():
                seg_head = SegHead(
                    in_channels=decoder.out_channels,
                    out_channels=n_classes,
                    kernel_size=1,
                    excitation_channels=head_excitation_channels,
                )
                self.add_module(f"{decoder_name}-{output_name}_head", seg_head)

    def forward_features(
        self, feats: List[torch.Tensor], style: torch.Tensor = None
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward all the decoders and return multi-res feature-lists per branch."""
        res = {}
        decoders = [k for k in self.keys() if "decoder" in k]

        for dec in decoders:
            featlist = self[dec](*feats, style=style)
            branch = "_".join(dec.split("_")[:-1])
            res[branch] = featlist

        return res

    def forward_heads(
        self, dec_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass all the seg heads."""
        res = {}
        heads = [k for k in self.keys() if "head" in k]
        for head in heads:
            branch_head = head.split("-")
            branch = branch_head[0]  # branch name
            head_name = "_".join(branch_head[1].split("_")[:-1])  # head name
            x = self[head](dec_feats[branch][-1])  # the last decoder stage feat map

            if self.out_size is not None:
                x = F.interpolate(
                    x, size=self.out_size, mode="bilinear", align_corners=False
                )

            res[f"{branch}-{head_name}"] = x

        return res

    def forward_style(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward the style domain adaptation layer."""
        style = None
        if self.make_style is not None:
            style = self.make_style(feat)

        return style

    def forward_stem_skip(
        self, x: torch.Tensor, dec_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward the stem skip connection."""
        stems = [k for k in self.keys() if "stem_skip" in k]
        for stem in stems:
            branch = stem.split("_")[0]
            dec_feats[branch][-1] = self[stem](x, dec_feats[branch][-1])

        return dec_feats

    def forward(
        self, enc_feats: Tuple[torch.Tensor, ...], x_in: torch.Tensor = None
    ) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
        """Forward pass style, decoders and optional stem skip.

        Parameters:
            enc_feats (Tuple[torch.Tensor, ...]):
                Tuple containing encoder feature tensors. Assumes that the deepest i.e.
                the bottleneck features is the last element of the tuple.
            x_in (torch.Tensor, default=None):
                Optional (the input image) tensor for stem skip connection.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
                The output of the seg heads.
        """
        enc_feats = enc_feats[::-1]  # bottleneck first
        if self.encoder_upsampler is not None:
            enc_feats = self.encoder_upsampler(enc_feats)

        style = self.forward_style(enc_feats[0])
        dec_feats = self.forward_features(enc_feats, style)

        # final input resolution skip connection
        if self.has_stem_skip and x_in is not None:
            dec_feats = self.forward_stem_skip(x_in, dec_feats)

        out = self.forward_heads(dec_feats)

        return enc_feats, dec_feats, out

    def initialize(self) -> None:
        """Initialize the decoders and segmentation heads."""
        for name, module in self.items():
            if "decoder" in name:
                initialize_decoder(module)
            if "head" in name:
                initialize_head(module)

    def _get_inner_keys(self, d: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get the inner dict keys from a nested dict."""
        return list(chain.from_iterable(list(d[k].keys()) for k in d.keys()))

    def _flatten_inner_dicts(self, d: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get the inner dicts as one dict from a nested dict."""
        return dict(chain.from_iterable(list(d[k].items()) for k in d.keys()))

    def _check_string_arg(self, arg: str) -> None:
        """Check the str arg does not contain any chars other than '_' for splitting."""
        if "-" in arg:
            raise ValueError(
                f"The dict key '{arg}' contains '-', which is not allowed. Use '_' instead."
            )

    def _check_decoder_args(self, decoders: Tuple[str, ...]) -> str:
        """Check `decoders` arg."""
        if len(decoders) != len(set(decoders)):
            raise ValueError("The decoder names need to be unique.")

        for dec in decoders:
            self._check_string_arg(dec)

    def _check_head_args(
        self, heads: Dict[str, int], decoders: Tuple[str, ...]
    ) -> None:
        """Check `heads` arg."""
        for head in heads.keys():
            self._check_string_arg(head)

        for head in self._get_inner_keys(heads):
            if head not in ALLOWED_HEADS:
                raise ValueError(
                    f"Unknown head type: '{head}'. Allowed: {ALLOWED_HEADS}."
                )

        if not set(decoders) == set(heads.keys()):
            raise ValueError(
                "The decoder names need match exactly to the keys of `heads`. "
                f"Got decoders: {decoders} and heads: {list(heads.keys())}."
            )

    def _check_depth(self, depth: int, arrs: Dict[str, Tuple[Any, ...]]) -> None:
        """Check that the depth matches to tuple args."""
        if not 3 <= depth <= 5:
            raise ValueError(
                f"max value for `depth` is 5, min value is 3. Got: {depth}"
            )

        for name, arr in arrs.items():
            if depth != len(arr):
                raise ValueError(
                    f"The length of `{name}` should be equal to arg `depth`: {depth}. "
                    f"For `{name}`, got: {arr}."
                )
