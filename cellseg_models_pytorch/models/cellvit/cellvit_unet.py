from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import (
    DecoderSoftOutput,
    MultiTaskDecoder,
)
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.cellvit._conf import _create_cellvit_args

__all__ = [
    "CellVitSamUnet",
    "cellvit_nuclei",
]


class CellVitSamUnet(nn.ModuleDict):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        out_channels: Tuple[int, ...] = (512, 256, 128, 64),
        layer_depths: Tuple[int, ...] = (3, 2, 2, 2),
        style_channels: int = None,
        enc_name: str = "samvit_base_patch16",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_channels: Tuple[int, ...] = None,
        enc_out_indices: Tuple[int, ...] = None,
        long_skip: str = "unet",
        merge_policy: str = "cat",
        short_skip: str = "basic",
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = True,
        attention: str = None,
        preattend: bool = False,
        out_size: int = None,
        encoder_kws: Dict[str, Any] = None,
        skip_kws: Dict[str, Any] = None,
        stem_skip_kws: Dict[str, Any] = None,
        inst_key: str = "inst",
        **kwargs,
    ) -> None:
        """CellVit-SAM implementation.

        CellVit:
            - https://arxiv.org/abs/2306.15350

        Args:
            decoders:
                Names of the decoder branches of this network. E.g. ("cellvit", "sem")
            heads:
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                of output name - number of output classes. E.g.
                {"cellvit": {"hovernet": 2}, "sem": {"sem": 5}, "type": {"type": 5}}
            depth (int):
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
                Defaults to 4.
            out_channels: Out channels for each decoder stage.
                Defaults to (512, 256, 64, 64).
            layer_depths: The number of conv blocks at each decoder stage.
                Defaults to (3, 2, 2, 2).
            style_channels:
                Number of style vector channels. If None, style vectors are ignored.
                Defaults to None.
            enc_name: Name of the encoder. See timm docs for more info.
                Defaults to "samvit_base_patch16".
            enc_pretrain: Whether to use imagenet pretrained weights in the encoder.
                Defaults to True.
            enc_freeze: Freeze encoder weights for training.
                Defaults to False.
            enc_out_channels: Out channels for each SAM-UnetTR encoder stage.
                Defaults to None.
            enc_out_indices:
                Indices of the encoder output features. If None, indices is set to
                `range(len(depth))`.
            upsampling (str):
                The upsampling method to be used. One of: "fixed-unpool", "nearest",
                "bilinear", "bicubic", "conv_transpose"
                Defaults to "fixed-unpool".
            long_skip:
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
                Defaults to "unet".
            merge_policy: The long skip merge policy. One of: "sum", "cat"
                Defaults to "sum".
            short_skip:
                The name of the short skip method. One of: "residual", "dense", "basic"
                Defaults to "basic".
            normalization: Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
                Defaults to "bn".
            activation: Activation method.
                One of: "mish", "swish", "relu", "relu6", "rrelu", "selu",
                "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
                Defaults to "relu".
            convolution:
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
                Defaults to "conv".
            preactivate: If True, normalization will be applied before convolution.
                Defaults to True.
            attention: Attention method. One of: "se", "scse", "gc", "eca", None
                Defaults to None.
            preattend:
                If True, Attention is applied at the beginning of forward pass.
                Defaults to False.
            out_size:
                If specified, the output size of the model will be (out_size, out_size).
                I.e. the outputs will be interpolated to this size.
            encoder_kws:
                Extra keyword arguments for the encoder. See timm docs for more info.
                Defaults to None.
            skip_kws: Extra keyword arguments for the skip-connection module.
                Defaults to None.
            stem_skip_kws:
                Extra keyword arguments for the stem skip-connection module.
                Defaults to None.
            inst_key: The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
                Defaults to "inst".
        """
        super().__init__()
        allowed = (
            "samvit_base_patch16",
            "samvit_base_patch16_224",
            "samvit_huge_patch16",
            "samvit_large_patch16",
        )
        if enc_name not in allowed:
            raise ValueError(
                f"Wrong encoder name. Got: {enc_name}. "
                f"Allowed encoder for CellVit: {allowed}"
            )

        self.inst_key = inst_key
        self.aux_key = "hovernet"
        self.depth = len(layer_depths)
        self.enc_name = enc_name

        if enc_out_indices is None:
            enc_out_indices = tuple(range(self.depth))

        if enc_out_channels is None:
            enc_out_channels = out_channels

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create decoder build args
        n_layers = (1,) * self.depth
        n_blocks = tuple([(d,) for d in layer_depths])

        stage_kws = _create_cellvit_args(
            layer_depths,
            normalization,
            activation,
            convolution,
            attention,
            preactivate,
            preattend,
            short_skip,
            use_style,
            merge_policy,
            skip_kws,
        )

        # set encoders
        self.add_module(
            self.enc_name,
            Encoder(
                timm_encoder_name=enc_name,
                timm_encoder_out_indices=enc_out_indices,
                timm_encoder_pretrained=enc_pretrain,
                timm_extra_kwargs=encoder_kws,
            ),
        )

        self.decoder = MultiTaskDecoder(
            decoders=decoders,
            heads=heads,
            out_channels=out_channels,
            enc_feature_info=self[self.enc_name].feature_info,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stage_kws=stage_kws,
            stem_skip_kws=stem_skip_kws,
            long_skip=long_skip,
            out_size=out_size,
            style_channels=style_channels,
        )

        # init decoder weights
        self.decoder.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self[self.enc_name].freeze_encoder()

        self.name = f"CellVit-{enc_name}"

    def forward(self, x: torch.Tensor, return_pred_only: bool = True) -> Dict[str, Any]:
        """Forward pass of Cellpose U-net.

            x (torch.Tensor):
                Input image batch. Shape: (B, C, H, W).
            return_pred_only (bool, default=True):
                If True, only the dense prediction maps are returned. If False, the
                encoder features and decoder features are also returned.


        Args:
            Returns (Dict[str, Any]):
                The output dictionary of the model. The keys of the dict are:
                    - "nuc": SoftInstanceOutput(type_map, aux_map, Optional[binary_map]).
                    - "cyto": SoftInstanceOutput(type_map, aux_map, Optional[binary_map]).
                    - "tissue": SoftSemanticOutput(type_map, Optional[binary_map]).
                    - "enc_feats": List[torch.Tensor].
                    - "dec_feats": Dict[str, List[torch.Tensor]].
                    - "enc_out": torch.Tensor.
        """
        enc_output, feats = self[self.enc_name](x)
        dec_out: DecoderSoftOutput = self.decoder(feats, x)

        res = {
            "nuc": dec_out.nuc_map,
            "tissue": dec_out.tissue_map,
            "cyto": dec_out.cyto_map,
        }

        if not return_pred_only:
            res["enc_feats"] = dec_out.enc_feats
            res["dec_feats"] = dec_out.dec_feats
            res["enc_out"] = enc_output

        return res


def cellvit_nuclei(enc_name: str, n_nuc_classes: int, **kwargs) -> nn.Module:
    """Initialaize CellVit for nuclei segmentation.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Args:
        enc_name:
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_nuc_classes: Number of nuclei type classes.
        **kwargs: Arbitrary key word args for the CellVitSAM class.

    Returns:
        nn.Module: The initialized CellVitSAM model.
    """
    cellvit_sam = CellVitSamUnet(
        enc_name=enc_name,
        decoders=("hovernet", "type"),
        heads={
            "hovernet": {"nuc_hovernet": 2},
            "type": {"nuc_type": n_nuc_classes},
        },
        **kwargs,
    )

    return cellvit_sam
