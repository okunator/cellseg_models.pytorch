from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import MultiTaskDecoder
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.cellpose._conf import _create_cellpose_args

__all__ = [
    "CellPoseUnet",
    "cellpose_base",
    "cellpose_plus",
    "omnipose_base",
    "omnipose_plus",
]


class CellPoseUnet(nn.Module):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        layer_depths: Tuple[int, ...] = (4, 4, 4, 4),
        style_channels: int = 256,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_indices: Tuple[int, ...] = None,
        upsampling: str = "fixed-unpool",
        long_skip: str = "unet",
        merge_policy: str = "sum",
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
        inst_key: str = "type",
        **kwargs,
    ) -> None:
        """Cellpose/Omnipose (2D) U-net model implementation.

        Cellpose:
            - https://www.nature.com/articles/s41592-020-01018-x

        Omnipose:
            - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2


        Note:
            Minor differences from the original implementation.
            - Different encoder, (any encoder from timm-library).
            - In the original implementation, all the outputs originate from one head,
              here each output has a distinct segmentation head.

        Parameters:
            decoders (Tuple[str, ...]):
                Names of the decoder branches of this network. E.g. ("cellpose", "sem")
            heads (Dict[str, Dict[str, int]]):
                Names of the decoder branches (has to match `decoders`) mapped to dicts
                 of output name - number of output classes. E.g.
                {"cellpose": {"type": 4, "cellpose": 2}, "sem": {"sem": 5}}
            depth (int, default=4):
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels (Tuple[int, ...], default=(256, 128, 64, 32)):
                Out channels for each decoder stage.
            layer_depths (Tuple[int, ...], default=(4, 4, 4, 4)):
                The number of conv blocks at each decoder stage.
            style_channels (int, default=256):
                Number of style vector channels. If None, style vectors are ignored.
            enc_name (str, default="resnet50"):
                Name of the encoder. See timm docs for more info.
            enc_pretrain (bool, default=True):
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze (bool, default=False):
                Freeze encoder weights for training.
            enc_out_indices (Tuple[int, ...], default=None):
                Indices of the output features from the encoder. If None, indices are
                set to `range(len(depth))`
            upsampling (str, default="fixed-unpool"):
                The upsampling method. One of: "fixed-unpool", "bilinear", "nearest",
                "conv_transpose", "bicubic"
            long_skip (str, default="unet"):
                long skip method. One of: "unet", "unetpp", "unet3p", "unet3p-lite", None
            merge_policy (str, default="sum"):
                The long skip merge policy. One of: "sum", "cat"
            short_skip (str, default="basic"):
                The name of the short skip method. One of: "residual", "dense", "basic"
            normalization (str, default="bn"):
                Normalization method. One of: "bn", "bcn", "gn", "in", "ln", None
            activation (str, default="relu"):
                Activation method. One of: "mish", "swish", "relu", "relu6", "rrelu",
                "selu", "celu", "gelu", "glu", "tanh", "sigmoid", "silu", "prelu",
                "leaky-relu", "elu", "hardshrink", "tanhshrink", "hardsigmoid"
            convolution (str, default="conv"):
                The convolution method. One of: "conv", "wsconv", "scaled_wsconv"
            preactivate (bool, default=True):
                If True, normalization will be applied before convolution.
            attention (str, default=None):
                Attention method. One of: "se", "scse", "gc", "eca", None
            preattend (bool, default=False):
                If True, Attention is applied at the beginning of forward pass.
            out_size (int, default=None):
                If specified, the output size of the model will be (out_size, out_size).
                I.e. the outputs will be interpolated to this size.
            encoder_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the encoder. See timm docs for more info.
            skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the skip-connection module.
            stem_skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the stem skip-connection module.
            inst_key (str, default="type"):
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
        """
        super().__init__()
        self.inst_key = inst_key
        self.aux_key = "cellpose"

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create build args
        n_layers = (1,) * depth
        n_blocks = tuple([(d,) for d in layer_depths])
        stage_kws = _create_cellpose_args(
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
            upsampling,
        )

        # set encoder
        self.encoder = Encoder(
            timm_encoder_name=enc_name,
            timm_encoder_out_indices=enc_out_indices,
            pixel_decoder_out_channels=out_channels,
            timm_encoder_pretrained=enc_pretrain,
            timm_extra_kwargs=encoder_kws,
        )

        # get the reduction factors for the encoder
        enc_reductions = tuple([inf["reduction"] for inf in self.encoder.feature_info])

        self.decoder = MultiTaskDecoder(
            decoders=decoders,
            heads=heads,
            out_channels=out_channels,
            enc_channels=self.encoder.out_channels,
            enc_reductions=enc_reductions,
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

        self.name = f"CellPoseUnet-{enc_name}"

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of Cellpose U-net.

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
        dec_feats, out = self.decoder.forward(feats, x)

        if return_feats:
            return enc_output, feats, dec_feats, out

        return out


def cellpose_base(n_type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline Cellpose U-net from kwargs.

    Cellpose:
    - https://www.nature.com/articles/s41592-020-01018-x

    Parameters:
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns:
        nn.Module: The initialized Cellpose U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("cellpose",),
        heads={"cellpose": {"cellpose": 2, "type": n_type_classes}},
        **kwargs,
    )

    return cellpose_unet


def cellpose_plus(n_type_classes: int, n_sem_classes: int, **kwargs) -> nn.Module:
    """Create the Cellpose U-net with a semantic decoder-branch from kwargs.

    Cellpose:
    - https://www.nature.com/articles/s41592-020-01018-x

    Parameters
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns:
        nn.Module: The initialized Cellpose+ U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("cellpose", "sem"),
        heads={
            "cellpose": {"cellpose": 2, "type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        **kwargs,
    )

    return cellpose_unet


def omnipose_base(n_type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline Omnipose U-net from kwargs.

    Omnipose:
    - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters:
        n_type_classes (int):
            Number of cell type classes in the dataset.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns:
        nn.Module: The initialized Cellpose U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("omnipose",),
        heads={"omnipose": {"omnipose": 2, "type": n_type_classes}},
        **kwargs,
    )
    cellpose_unet.aux_key = "omnipose"

    return cellpose_unet


def omnipose_plus(n_type_classes: int, n_sem_classes: int, **kwargs) -> nn.Module:
    """Create the Omnipose U-net with a semantic decoder-branch from kwargs.

    Omnipose:
    - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters:
        n_type_classes (int):
            Number of cell type classes in the dataset.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns:
        nn.Module: The initialized Cellpose+ U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("omnipose", "sem"),
        heads={
            "omnipose": {"omnipose": 2, "type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        **kwargs,
    )
    cellpose_unet.aux_key = "omnipose"

    return cellpose_unet
