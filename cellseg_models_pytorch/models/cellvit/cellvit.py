from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import MultiTaskDecoder
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.cellvit._conf import _create_cellvit_args

__all__ = [
    "CellVitSAM",
    "cellvit_sam_base",
    "cellvit_sam_small",
    "cellvit_sam_plus",
    "cellvit_sam_small_plus",
]


class CellVitSAM(nn.Module):
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

        Parameters:
            decoders : Tuple[str, ...]
                Names of the decoder branches of this network. E.g. ("cellvit", "sem")
            heads : Dict[str, Dict[str, int]]
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                of output name - number of output classes. E.g.
                {"cellvit": {"hovernet": 2}, "sem": {"sem": 5}, "type": {"type": 5}}
            depth : int, default=4
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels : Tuple[int, ...], default=(512, 256, 64, 64)
                Out channels for each decoder stage.
            layer_depths : Tuple[int, ...], default=(3, 2, 2, 2)
                The number of conv blocks at each decoder stage.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
            enc_name : str, default="samvit_base_patch16"
                Name of the encoder. See timm docs for more info.
            enc_pretrain : bool, default=True
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze : bool, default=False
                Freeze encoder weights for training.
            enc_out_channels : Tuple[int, ...], default=None
                Out channels for each SAM-UnetTR encoder stage.
            enc_out_indices : Tuple[int, ...], optional
                Indices of the encoder output features. If None, indices is set to
                `range(len(depth))`.
            upsampling : str, default="fixed-unpool"
                The upsampling method to be used. One of: "fixed-unpool", "nearest",
                "bilinear", "bicubic", "conv_transpose"
            long_skip : str, default="unet"
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            merge_policy : str, default="sum"
                The long skip merge policy. One of: "sum", "cat"
            short_skip : str, default="basic"
                The name of the short skip method. One of: "residual", "dense", "basic"
            normalization : str, default="bn":
                Normalization method.
                One of: "bn", "bcn", "gn", "in", "ln", None
            activation : str, default="relu"
                Activation method.
                One of: "mish", "swish", "relu", "relu6", "rrelu", "selu",
                "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
            convolution : str, default="conv"
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            preactivate : bool, default=True
                If True, normalization will be applied before convolution.
            attention : str, default=None
                Attention method. One of: "se", "scse", "gc", "eca", None
            preattend : bool, default=False
                If True, Attention is applied at the beginning of forward pass.
            out_size : int, optional
                If specified, the output size of the model will be (out_size, out_size).
                I.e. the outputs will be interpolated to this size.
            encoder_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the encoder. See timm docs for more info.
            skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the skip-connection module.
            stem_skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the stem skip-connection module.
            inst_key : str, default="inst"
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
        """
        super().__init__()
        self.inst_key = inst_key
        self.aux_key = "hovernet"
        self.depth = len(layer_depths)

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

        # set encoders
        self.encoder = Encoder(
            timm_encoder_name=enc_name,
            timm_encoder_out_indices=enc_out_indices,
            timm_encoder_pretrained=enc_pretrain,
            timm_extra_kwargs=encoder_kws,
        )

        self.decoder = MultiTaskDecoder(
            decoders=decoders,
            heads=heads,
            out_channels=out_channels,
            enc_feature_info=self.encoder.feature_info,
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
            self.encoder.freeze_encoder()

        self.name = f"CellVit-{enc_name}"

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of Hover-Net.

        Parameters:
            x (torch.Tensor):
                Input image batch. Shape: (B, C, H, W).
            return_feats (bool, default=False):
                If True, encoder, decoder, and head outputs will all be returned

        Returns:
            Dict[str, torch.Tensor]:
                Dictionary of outputs. if `return_feats == True` returns also the encoder
                output, a list of encoder features, dict of decoder features and the head
                outputs (segmentations) dict.
        """
        enc_output, feats = self.encoder.forward(x)
        feats, dec_feats, out = self.decoder.forward(feats, x)

        if return_feats:
            return enc_output, feats, dec_feats, out

        return out


def cellvit_sam_base(enc_name: str, n_type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline CellVit-SAM (three decoders) from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters:
        enc_name (str):
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns:
        nn.Module: The initialized CellVitSAM model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "inst"),
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
            "inst": {"inst": 2},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_plus(
    enc_name: str, n_type_classes: int, n_sem_classes: int, **kwargs
) -> nn.Module:
    """Create CellVit-SAM (additional semantic decoders) from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters:
        enc_name (str):
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns:
        nn.Module: The initialized CellVitSAM+ model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "inst", "sem"),
        heads={
            "hovernet": {"hovernet": 2},
            "sem": {"sem": n_sem_classes},
            "type": {"type": n_type_classes},
            "inst": {"inst": 2},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_small(enc_name: str, n_type_classes: int, **kwargs) -> nn.Module:
    """Create CellVit-SAM without inst decoder branch from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters:
        enc_name (str):
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns:
        nn.Module: The initialized CellVitSAM-small model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_small_plus(
    enc_name: str, n_type_classes: int, n_sem_classes: int, **kwargs
) -> nn.Module:
    """Create the CellVit-SAM+ without inst decoder branch from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters:
        enc_name (str):
            Name of the encoder. One of: "samvit_base_patch16", "samvit_base_patch16_224",
            "samvit_huge_patch16", "samvit_large_patch16"
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellVitsSAM class.

    Returns:
        nn.Module: The initialized CellVit-SAM-small+ model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "sem"),
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        inst_key="type",
        **kwargs,
    )

    return cellvit_sam
