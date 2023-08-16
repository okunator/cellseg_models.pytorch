from itertools import chain
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ._initialization import initialize_decoder, initialize_head

__all__ = ["BaseMultiTaskSegModel"]


class BaseMultiTaskSegModel(nn.ModuleDict):
    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward pass for encoder, style and decoders.

        NOTE: Returns both encoder and decoder features, not style.
        """
        feats = self.forward_encoder(x)
        style = self.forward_style(feats[0])
        dec_feats = self.forward_dec_features(feats, style)

        # final input resolution skip connection
        if self.add_stem_skip:
            dec_feats = self.forward_stem_skip(x, dec_feats)

        return feats, dec_feats

    def forward_stem_skip(
        self, x: torch.Tensor, dec_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward the stem skip connection."""
        stems = [k for k in self.keys() if "stem_skip" in k]
        for stem in stems:
            branch = stem.split("_")[0]
            dec_feats[branch][-1] = self[stem](x, dec_feats[branch][-1])

        return dec_feats

    def forward_encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward the model encoder."""
        self._check_input_shape(x)
        feats = self.encoder(x)

        return feats

    def forward_style(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward the style domain adaptation layer.

        NOTE: returns None if style channels are not given at model init.
        """
        style = None
        if self.make_style is not None:
            style = self.make_style(feat)

        return style

    def forward_dec_features(
        self, feats: List[torch.Tensor], style: torch.Tensor = None
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass of all the decoder features mappings in the model.

        NOTE: returns all the features from diff decoder stages in a list.
        """
        res = {}
        decoders = [k for k in self.keys() if "decoder" in k]

        for dec in decoders:
            featlist = self[dec](*feats, style=style)
            branch = dec.split("_")[0]
            res[branch] = featlist

        return res

    def forward_heads(
        self, dec_feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the seg heads in a multi-task seg model."""
        res = {}
        heads = [k for k in self.keys() if "head" in k]
        for head in heads:
            branch = head.split("_")[0]
            x = self[head](dec_feats[branch][-1])  # the last decoder stage feat map
            res[branch] = x

        return res

    def initialize(self) -> None:
        """Init the decoder branches and their classification/regression heads."""
        for name, module in self.items():
            if "decoder" in name:
                initialize_decoder(module)
            if "head" in name:
                initialize_head(module)

    def freeze_encoder(self) -> None:
        """Freeze the parameters of the encoeder."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _check_input_shape(self, x: torch.Tensor) -> None:
        """Check that the input is divisible by 32."""
        h, w = x.shape[-2:]

        if (h % 32) + (w % 32):
            raise RuntimeError(
                "Illegal input shape. Expected input H/W to be divisible by 32. "
                f"Got height: {h}, and width: {w}."
            )

    def _check_decoder_args(
        self, decoders: Tuple[str], must_haves: Tuple[str, ...]
    ) -> str:
        """Check that the decoders arg contains needed values."""
        if not any([d in must_haves for d in decoders]):
            raise ValueError(
                f"`decoders` need to contain one of: {must_haves} " f"Got: {decoders}."
            )

        if len(must_haves) > 1:
            if all([m in decoders for m in must_haves]):
                raise ValueError(
                    f"`decoders` need to contain only one of: {must_haves} "
                    f"Got: {decoders}."
                )

        for must_have in must_haves:
            try:
                ix = decoders.index(must_have)
                return decoders[ix]
            except ValueError:
                pass

    def _get_inner_keys(self, d: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get the inner dict keys from a nested dict."""
        return list(chain.from_iterable(list(d[k].keys()) for k in d.keys()))

    def _flatten_inner_dicts(self, d: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get the inner dicts as one dict from a nested dict."""
        return dict(chain.from_iterable(list(d[k].items()) for k in d.keys()))

    def _check_head_args(
        self, heads: Dict[str, int], decoders: Tuple[str, ...]
    ) -> None:
        """Check for illegal `heads` args."""
        if not set(decoders) == set(heads.keys()):
            raise ValueError(
                "The decoder names need match exactly to the keys of `heads`. "
                f"Got decoders: {decoders} and heads: {list(heads.keys())}."
            )

        for head in heads.keys():
            if not any([h == head for h in heads[head].keys()]):
                raise ValueError(
                    "For every decoder name one matching head name has to exist "
                    f"Got heads: {[j for h, i in heads.items() for j in i.keys()]}"
                    f"decoders: {decoders}."
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
