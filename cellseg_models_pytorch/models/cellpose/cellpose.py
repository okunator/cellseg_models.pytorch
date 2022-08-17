from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...decoders import Decoder
from ...modules.misc_modules import StyleReshape
from ..base._base_model import BaseMultiTaskSegModel
from ..base._seg_head import SegHead
from ..base._timm_encoder import TimmEncoder
from ._conf import _create_cellpose_args

__all__ = [
    "CellPoseUnet",
    "cellpose_base",
    "cellpose_plus",
    "omnipose_base",
    "omnipose_plus",
]


class CellPoseUnet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str],
        heads: Dict[str, Dict[str, int]],
        inst_key: str = "type",
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        layer_depths: Tuple[int, ...] = (4, 4, 4, 4),
        style_channels: int = 256,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        long_skip: str = "unet",
        merge_policy: str = "sum",
        short_skip: str = "basic",
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = True,
        attention: str = None,
        preattend: bool = False,
        skip_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Cellpose/Omnipose (2D) U-net model implementation.

        Cellpose:
            - https://www.nature.com/articles/s41592-020-01018-x

        Omnipose:
            - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

                      (|------ SEMANTIC_DECODER --- SEMANTIC_HEAD)
        ENCODER -------|
                       |                         | --- FLOWS_HEAD
                       |------ FLOWS_DECODER ----|
                                                (| --- INSTANCE/TYPE_HEAD)

        NOTE: Minor differences from the original implementation.
        - Different encoder, (any encoder from timm-library).
        - In the original implementation, all the outputs originate from
            one head, here each output has a distinct segmentation head.

        Parameters
        ----------
            decoders : Tuple[str]
                Names of the decoder branches of this network. E.g. ("cellpose", "sem")
            heads : Dict[str, Dict[str, int]]
                Names of the decoder branches (has to match `decoders`) mapped to dicts
                 of output name - number of output classes. E.g.
                {"cellpose": {"type": 4, "cellpose": 2}, "sem": {"sem": 5}}
            inst_key : str, default="type"
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
            depth : int, default=4
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels : Tuple[int, ...], default=(256, 128, 64, 32)
                Out channels for each decoder stage.
            layer_depths : Tuple[int, ...], default=(4, 4, 4, 4)
                The number of conv blocks at each decoder stage.
            style_channels : int, default=256
                Number of style vector channels. If None, style vectors are ignored.
            enc_name : str, default="resnet50"
                Name of the encoder. See timm docs for more info.
            enc_pretrain : bool, default=True
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze : bool, default=False
                Freeze encoder weights for training.
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
            skip_params : Optional[Dict]
                Extra keyword arguments for the skip-connection module. These depend
                on the skip module. Refer to specific skip modules for more info.

        Raises
        ------
            ValueError: If `decoders` does not contain either 'cellpose' or 'omnipose'.
            ValueError: If `decoders` contain both 'cellpose' and 'omnipose'.
            ValueError: If `heads` keys don't match `decoders`.
            ValueError: If decoder names don't have a matching head name in `heads`.
        """
        super().__init__()
        self.aux_key = self._check_decoder_args(decoders, ("omnipose", "cellpose"))
        self.inst_key = inst_key
        self._check_head_args(heads, decoders)
        self._check_depth(
            depth, {"out_channels": out_channels, "layer_depths": layer_depths}
        )

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create build args
        n_layers = (1,) * depth
        n_blocks = tuple([(d,) for d in layer_depths])
        dec_params = {
            d: _create_cellpose_args(
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
                skip_params,
            )
            for d in decoders
        }

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
                out_channels=out_channels,
                style_channels=style_channels,
                long_skip=long_skip,
                n_layers=n_layers,
                n_blocks=n_blocks,
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

        self.name = f"CellPoseUnet-{enc_name}"

        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of Cellpose U-net."""
        self._check_input_shape(x)

        feats = self.encoder(x)

        style = None
        if self.make_style is not None:
            style = self.make_style(feats[0])

        dec_feats = self.forward_dec_features(feats, style)

        for decoder_name in self.heads.keys():
            for head_name in self.heads[decoder_name].keys():
                k = self.aux_key if head_name not in dec_feats.keys() else head_name
                dec_feats[head_name] = dec_feats[k]

        out = self.forward_heads(dec_feats)

        return out


def cellpose_base(type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline Cellpose U-net from kwargs.

    Cellpose:
    - https://www.nature.com/articles/s41592-020-01018-x

    Parameters
    ----------
        type_classes : int
            Number of type classes in the dataset.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns
    -------
        nn.Module: The initialized Cellpose U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("cellpose",),
        heads={"cellpose": {"cellpose": 2, "type": type_classes}},
        **kwargs,
    )

    return cellpose_unet


def cellpose_plus(type_classes: int, sem_classes: int, **kwargs) -> nn.Module:
    """Create the Cellpose U-net with a semantic decoder-branch from kwargs.

    Cellpose:
    - https://www.nature.com/articles/s41592-020-01018-x

    Parameters
    ----------
        type_classes : int
            Number of type classes in the dataset.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns
    -------
        nn.Module: The initialized Cellpose+ U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("cellpose", "sem"),
        heads={
            "cellpose": {"cellpose": 2, "type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return cellpose_unet


def omnipose_base(type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline Omnipose U-net from kwargs.

    Omnipose:
    - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters
    ----------
        type_classes : int
            Number of type classes in the dataset.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns
    -------
        nn.Module: The initialized Cellpose U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("omnipose",),
        heads={"omnipose": {"omnipose": 2, "type": type_classes}},
        **kwargs,
    )

    return cellpose_unet


def omnipose_plus(type_classes: int, sem_classes: int, **kwargs) -> nn.Module:
    """Create the Omnipose U-net with a semantic decoder-branch from kwargs.

    Omnipose:
    - https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2

    Parameters
    ----------
        type_classes : int
            Number of type classes in the dataset.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the CellPoseUnet class.

    Returns
    -------
        nn.Module: The initialized Cellpose+ U-net model.
    """
    cellpose_unet = CellPoseUnet(
        decoders=("omnipose", "sem"),
        heads={
            "omnipose": {"omnipose": 2, "type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return cellpose_unet
