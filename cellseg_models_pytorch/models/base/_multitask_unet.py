from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml

from ...decoders import Decoder
from ...decoders.long_skips import StemSkip
from ...encoders import Encoder
from ...modules.misc_modules import StyleReshape
from ._base_model import BaseMultiTaskSegModel
from ._seg_head import SegHead

__all__ = ["MultiTaskUnet"]


class MultiTaskUnet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        long_skips: Dict[str, Union[str, Tuple[str, ...]]],
        out_channels: Dict[str, Tuple[int, ...]],
        n_conv_layers: Dict[str, Tuple[int, ...]] = None,
        n_conv_blocks: Dict[str, Tuple[Tuple[int, ...], ...]] = None,
        n_transformers: Dict[str, Tuple[int, ...]] = None,
        n_transformer_blocks: Dict[str, Tuple[Tuple[int, ...], ...]] = None,
        dec_params: Dict[str, Tuple[Dict[str, Any], ...]] = None,
        depth: int = 4,
        style_channels: int = 256,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        inst_key: str = None,
        aux_key: str = None,
        add_stem_skip: bool = False,
        stem_params: Dict[str, Any] = None,
        encoder_params: Optional[Dict] = None,
        unettr_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Create a universal multi-task (2D) unet.

        NOTE: For experimental purposes.

        Parameters
        ----------
            decoders : Tuple[str, ...]
                Names of the decoder branches of this network. E.g. ("cellpose", "sem")
            heads : Dict[str, Dict[str, int]]
                Names of the decoder branches (has to match `decoders`) mapped to dicts
                 of output name - number of output classes. E.g.
                {"cellpose": {"type": 4, "cellpose": 2}, "sem": {"sem": 5}}
            out_channels : Tuple[int, ...]
                Out channels for each decoder stage.
            long_skips : Dict[str, str]
                Dictionary mapping decoder branch-names to tuples defining the long skip
                method to be used inside each of the decoder stages.
                Allowed: "cross-attn", "unet", "unetpp", "unet3p", "unet3p-lite", None
            n_conv_layers : Dict[str, Tuple[int, ...]], optional
                Dictionary mapping decoder branch-names to tuples defining the number of
                conv layers inside each of the decoder stages.
            n_conv_blocks : Dict[str, Tuple[Tuple[int, ...], ...]], optional
                The number of blocks inside each conv-layer in each decoder stage.
            n_transformers : Tuple[int, ...], optional
                Dictionary mapping decoder branch-names to tuples defining the number of
                transformer layers inside each of the decoder stages.
            n_transformer_blocks : Tuple[Tuple[int]], optional
                The number of transformer blocks inside each transformer-layer at each
                decoder stage.
            dec_params : Dict[str, Tuple[Dict[str, Any], ...]], optional
                The keyword args for each of the distinct decoder stages. Incudes the
                parameters for the long skip connections and convolutional layers of the
                decoder itself. See the `DecoderStage` documentation for more info.
            depth : int, default=4
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            style_channels : int, default=256
                Number of style vector channels. If None, style vectors are ignored.
            enc_name : str, default="resnet50"
                Name of the encoder. See timm docs for more info.
            enc_pretrain : bool, default=True
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze : bool, default=False
                Freeze encoder weights for training.
            inst_key : str, optional
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
            aux_key : str, optional
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the auxilliary map.
            add_stem_skip : bool, default=False
                If True, a stem conv block is added to the model whose output is used
                as a long skip input at the final decoder layer that is the highest
                resolution layer and the same resolution as the input image.
            stem_params : Dict[str, Any], optional
                The keyword args for the stem conv block. See `StemSkip` for more info.
            encoder_params : Optional[Dict]
                Extra keyword arguments for the encoder. These depend on the encoder.
                Refer to specific encoders for more info.
            unettr_kwargs : Dict[str, Any]
                Key-word arguments for the transformer encoder. These arguments are used
                only if the encoder is transformer based. Refer to the docstring of the
                `EncoderUnetTR`
        """
        super().__init__()
        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads
        self.decoders = decoders
        self.inst_key = inst_key
        self.aux_key = aux_key
        self.add_stem_skip = add_stem_skip

        # set encoder
        self.encoder = Encoder(
            enc_name,
            depth=depth,
            pretrained=enc_pretrain,
            checkpoint_path=kwargs.get("checkpoint_path", None),
            unettr_kwargs=unettr_kwargs,
            **encoder_params if encoder_params is not None else {},
        )

        # get the reduction factors for the encoder
        enc_reductions = tuple([inf["reduction"] for inf in self.encoder.feature_info])

        # style
        self.make_style = None
        if use_style:
            self.make_style = StyleReshape(self.encoder.out_channels[0], style_channels)

        # set decoders
        for decoder_name in decoders:
            decoder = Decoder(
                enc_channels=self.encoder.out_channels,
                enc_reductions=enc_reductions,
                out_channels=out_channels[decoder_name],
                long_skip=long_skips[decoder_name],
                n_conv_layers=self._kwarg(n_conv_layers)[decoder_name],
                n_conv_blocks=self._kwarg(n_conv_blocks)[decoder_name],
                n_transformers=self._kwarg(n_transformers)[decoder_name],
                n_transformer_blocks=self._kwarg(n_transformer_blocks)[decoder_name],
                style_channels=style_channels,
                stage_params=dec_params[decoder_name],
            )
            self.add_module(f"{decoder_name}_decoder", decoder)

        # optional stem skip
        if add_stem_skip:
            for decoder_name in decoders:
                stem_skip = StemSkip(
                    out_channels=decoder.out_channels,
                    n_blocks=2,
                    **stem_params if stem_params is not None else {},
                )
                self.add_module(f"{decoder_name}_stem_skip", stem_skip)

        # set heads
        for decoder_name in heads.keys():
            for output_name, n_classes in heads[decoder_name].items():
                seg_head = SegHead(
                    in_channels=decoder.out_channels,
                    out_channels=n_classes,
                    kernel_size=1,
                )
                self.add_module(f"{output_name}_seg_head", seg_head)

        self.name = f"MultiTaskUnet-{enc_name}"

        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    def _kwarg(self, kw: Union[None, Dict[str, Any]]) -> Dict[str, Any]:
        """Return a placeholder dict kwarg if `kw` is None. Else return `kw`."""
        if kw is None:
            kw = {d: None for d in self.decoders}
        return kw

    @classmethod
    def from_yaml(cls, yaml_path: str) -> nn.Module:
        """Initialize the multi-tasks U-net from a yaml-file.

        Parameters
        ----------
            yaml_path : str
                Path to the yaml file containing rest of the params
        """
        with open(yaml_path, "r") as stream:
            kwargs = yaml.full_load(stream)

        return cls(**kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of Multi-task U-net."""
        feats = self.forward_encoder(x)
        style = self.forward_style(feats[0])
        dec_feats = self.forward_dec_features(feats, style)

        for decoder_name in self.heads.keys():
            for head_name in self.heads[decoder_name].keys():
                k = self.aux_key if head_name not in dec_feats.keys() else head_name
                dec_feats[head_name] = dec_feats[k]

        out = self.forward_heads(dec_feats)

        return out
