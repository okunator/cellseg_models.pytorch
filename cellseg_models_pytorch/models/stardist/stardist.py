from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...decoders import Decoder
from ...modules.misc_modules import StyleReshape
from ..base._base_model import BaseMultiTaskSegModel
from ..base._seg_head import SegHead
from ..base._timm_encoder import TimmEncoder
from ._conf import _create_stardist_args

__all__ = ["StarDistUnet", "stardist_base", "stardist_base_multiclass", "stardist_plus"]


class StarDistUnet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        extra_convs: Dict[str, Dict[str, int]],
        heads: Dict[str, Dict[str, int]],
        inst_key: str = "dist",
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        long_skip: str = "unet",
        merge_policy: str = "cat",
        short_skip: str = "basic",
        block_type: str = "basic",
        normalization: str = None,
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = False,
        attention: str = None,
        preattend: bool = False,
        skip_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Stardist (2D) U-Net model implementation.

        Stardist:
            - https://arxiv.org/abs/1806.03535

                      (|------ SEMANTIC_DECODER --- CONV --- SEMANTIC_HEAD)
        ENCODER -------|
                       |                                    |---- RAY_HEAD
                       |------ STARDIST_DECODER --- CONV ---|
                                                 |          |---- DISTPROB_HEAD
                                                 |
                                                (|- CONV --- TYPE_HEAD)

        NOTE: Minor differences from the original implementation.
        - long skip concatenation/sum applied before each conv layer rather than after.

        Parameters
        ----------
            decoders : Tuple[str]
                Names of the decoder branches of this network. E.g. ("stardist", "sem")
            extra_convs : Dict[str, Dict[str, int]]
                The extra conv blocks before segmentation heads of the architecture.
                I.e. Names of the decoder branches (has to match `decoders`) mapped to
                dicts of output name - number of output channels. E.g.
                {"stardist": {"type": 128, "stardist": 128}, "sem": {"sem": 128}}
            heads : Dict[str, Dict[str, int]]
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                 of output name - number of output classes. E.g.
                {"cellpose": {"type": 4, "cellpose": 2}, "sem": {"sem": 5}}
            inst_key : str, default="dist"
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
            depth : int, default=4
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels : Tuple[int, ...], default=(256, 128, 64, 32)
                Out channels for each decoder stage.
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
            ValueError: If `decoders` does not contain 'stardist'.
            ValueError: If `extra_convs` keys don't match `decoders`.
            ValueError: If extra_convs names don't have a matching head name in `heads`.
        """
        super().__init__()
        self.aux_key = self._check_decoder_args(decoders, ("stardist",))
        self.inst_key = inst_key
        self._check_head_args(extra_convs, decoders)
        self._check_head_args(heads, self._get_inner_keys(extra_convs))
        self._check_depth(depth, {"out_channels": out_channels})

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.extra_convs = extra_convs
        self.heads = heads

        n_layers = (1,) * depth
        n_blocks = ((2,),) * depth
        dec_params = {
            d: _create_stardist_args(
                depth,
                normalization,
                activation,
                convolution,
                attention,
                preactivate,
                preattend,
                short_skip,
                use_style,
                block_type,
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

        # set decoder
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

        # set additional conv blocks ('avoid “fight over features”'.)
        for decoder_name in extra_convs.keys():
            for extra_conv, n_channels in extra_convs[decoder_name].items():
                features = nn.Conv2d(
                    in_channels=decoder.out_channels,
                    out_channels=n_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
                self.add_module(f"{extra_conv}_features", features)

        # set heads
        for decoder_name in extra_convs.keys():
            for extra_conv, in_channels in extra_convs[decoder_name].items():
                for output_name, n_classes in heads[extra_conv].items():
                    seg_head = SegHead(
                        in_channels=in_channels,
                        out_channels=n_classes,
                        kernel_size=1,
                    )
                    self.add_module(f"{output_name}_seg_head", seg_head)

        self.name = f"StardistUnet-{enc_name}"

        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of Stardist."""
        self._check_input_shape(x)

        feats = self.encoder(x)

        style = None
        if self.make_style is not None:
            style = self.make_style(feats[0])

        dec_feats = self.forward_dec_features(feats, style)
        # Extra convs after decoders
        for e in self.extra_convs.keys():
            for extra_conv in self.extra_convs[e].keys():
                k = self.aux_key if extra_conv not in dec_feats.keys() else extra_conv
                dec_feats[extra_conv] = self[f"{extra_conv}_features"](dec_feats[k])

        # seg heads
        for decoder_name in self.heads.keys():
            for head_name in self.heads[decoder_name].keys():
                k = self.aux_key if head_name not in dec_feats.keys() else head_name
                if k != head_name:
                    dec_feats[head_name] = dec_feats[k]

        out = self.forward_heads(dec_feats)

        return out


def stardist_base(n_rays: int, **kwargs) -> nn.Module:
    """Create the Stardist U-net from kwargs.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns
    -------
        nn.Module: The initialized Stardist U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist",),
        extra_convs={"stardist": {"stardist": 128}},
        heads={"stardist": {"stardist": n_rays, "dist": 1}},
        **kwargs,
    )

    return stardist_unet


def stardist_base_multiclass(n_rays: int, type_classes: int, **kwargs) -> nn.Module:
    """Create the Stardist U-net with an extra multi-class segmentation head.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        type_classes : int
            Number of type classes in the dataset.
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns
    -------
        nn.Module: The initialized multiclass Stardist U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist",),
        extra_convs={"stardist": {"type": 128, "stardist": 128}},
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": type_classes},
        },
        **kwargs,
    )

    return stardist_unet


def stardist_plus(
    n_rays: int, type_classes: int, sem_classes: int, **kwargs
) -> nn.Module:
    """Create the Stardist U-net with a semantic decoder-branch.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        type_classes : int
            Number of type classes in the dataset.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns
    -------
        nn.Module: The initialized Stardist+ U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist", "sem"),
        extra_convs={"stardist": {"type": 128, "stardist": 128}, "sem": {"sem": 128}},
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return stardist_unet
