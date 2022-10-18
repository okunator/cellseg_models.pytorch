from typing import Any, Dict, Tuple

import torch

from ...decoders import Decoder
from ...modules.misc_modules import StyleReshape
from ._base_model import BaseMultiTaskSegModel
from ._seg_head import SegHead
from ._timm_encoder import TimmEncoder

__all__ = ["MultiTaskUnet"]


class MultiTaskUnet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        n_layers: Dict[str, Tuple[int, ...]],
        n_blocks: Dict[str, Tuple[Tuple[int, ...], ...]],
        out_channels: Dict[str, Tuple[int, ...]],
        long_skips: Dict[str, str],
        dec_params: Dict[str, Tuple[Dict[str, Any], ...]],
        depth: int = 4,
        style_channels: int = 256,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        inst_key: str = None,
        aux_key: str = None,
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
            n_layers : Dict[str, Tuple[int, ...]]
                The number of conv layers inside each of the decoder stages.
            n_blocks : Dict[str, Tuple[Tuple[int, ...], ...]]
                The number of blocks inside each conv-layer in each decoder stage.
            out_channels : Tuple[int, ...]
                Out channels for each decoder stage.
            long_skips : Dict[str, str]
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            dec_params : Dict[str, Tuple[Dict[str, Any], ...]])
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
        """
        super().__init__()
        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads
        self.inst_key = inst_key
        self.aux_key = aux_key

        # set timm encoder
        self.encoder = TimmEncoder(
            enc_name,
            depth=depth,
            pretrained=enc_pretrain,
        )

        # style
        self.make_style = None
        if use_style:
            self.make_style = StyleReshape(self.encoder.out_channels[0], style_channels)

            # set decoders
        for decoder_name in decoders:
            decoder = Decoder(
                enc_channels=list(self.encoder.out_channels),
                style_channels=style_channels,
                out_channels=out_channels[decoder_name],
                long_skip=long_skips[decoder_name],
                n_layers=n_layers[decoder_name],
                n_blocks=n_blocks[decoder_name],
                stage_params=dec_params[decoder_name],
            )
            self.add_module(f"{decoder_name}_decoder", decoder)

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
