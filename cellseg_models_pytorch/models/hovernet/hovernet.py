from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders import UnetDecoder
from cellseg_models_pytorch.decoders.long_skips import StemSkip
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.modules.misc_modules import StyleReshape

from ..base._base_model import BaseMultiTaskSegModel
from ..base._seg_head import SegHead
from ._conf import _create_hovernet_args

__all__ = [
    "HoverNet",
    "hovernet_base",
    "hovernet_small",
    "hovernet_plus",
    "hovernet_small_plus",
]


class HoverNet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        inst_key: str = "inst",
        depth: int = 4,
        out_channels: Tuple[int, ...] = (512, 256, 64, 64),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        upsampling: str = "fixed-unpool",
        long_skip: str = "unet",
        merge_policy: str = "sum",
        n_dense: Tuple[int, int] = (8, 4),
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = True,
        attention: str = None,
        preattend: bool = False,
        add_stem_skip: bool = False,
        out_size: Optional[int] = None,
        skip_params: Optional[Dict] = None,
        encoder_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Hover-Net implementation.

        HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

                      (|------ SEMANTIC_DECODER ----- SEMANTIC_HEAD)
                       |
                       |------ TYPE_DECODER ---------- TYPE_HEAD
        ENCODER -------|
                       |------ HOVER_DECODER --------- HOVER_HEAD
                       |
                      (|------ INSTANCE_DECODER ---- INSTANCE_HEAD)

        NOTE: Minor differences from the original implementation.
        - Different encoder, (any encoder from timm-library).
        - Dense blocks have transition conv-blocks like in the original dense-net.

        Parameters
        ----------
            decoders : Tuple[str, ...]
                Names of the decoder branches of this network. E.g. ("hovernet", "sem")
            heads : Dict[str, Dict[str, int]]
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                of output name - number of output classes. E.g.
                {"hovernet": {"hovernet": 2}, "sem": {"sem": 5}, "type": {"type": 5}}
            inst_key : str, default="inst"
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
            depth : int, default=4
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels : Tuple[int, ...], default=(512, 256, 64, 64)
                Out channels for each decoder stage.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
            enc_name : str, default="resnet50"
                Name of the encoder. See timm docs for more info.
            enc_pretrain : bool, default=True
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze : bool, default=False
                Freeze encoder weights for training.
            upsampling : str, default="fixed-unpool"
                The upsampling method to be used. One of: "fixed-unpool", "nearest",
                "bilinear", "bicubic", "conv_transpose"
            long_skip : str, default="unet"
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            merge_policy : str, default="sum"
                The long skip merge policy. One of: "sum", "cat"
            n_dense : Tuple[int, int], default=(8, 4)
                Number of dense blocks in the dense decoder stages.
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
            add_stem_skip : bool, default=False
                If True, a stem conv block is added to the model whose output is used
                as a long skip input at the final decoder layer that is the highest
                resolution layer and the same resolution as the input image.
            out_size : int, optional
                If specified, the output size of the model will be (out_size, out_size).
                I.e. the outputs will be interpolated to this size.
            skip_params : Optional[Dict]
                Extra keyword arguments for the skip-connection modules. These depend
                on the skip module. Refer to specific skip modules for more info. I.e.
                `UnetSkip`, `UnetppSkip`, `Unet3pSkip`.
            encoder_params : Optional[Dict]
                Extra keyword arguments for the encoder. These depend on the encoder.
                Refer to specific encoders for more info.

        Raises
        ------
            ValueError: If `decoders` does not contain 'hovernet'.
            ValueError: If `heads` keys don't match `decoders`.
            ValueError: If decoder names don't have a matching head name in `heads`.
        """
        super().__init__()
        self.out_size = out_size
        self.aux_key = self._check_decoder_args(decoders, ("hovernet",))
        self.inst_key = inst_key
        self._check_head_args(heads, decoders)
        self._check_depth(depth, {"out_channels": out_channels})

        self.add_stem_skip = add_stem_skip
        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create decoder build args
        n_layers = (3, 3) + (1,) * (depth - 2)
        n_blocks = ((1, n_dense[0], 1), (1, n_dense[1], 1)) + ((1,),) * (depth - 2)
        dec_params = {
            d: _create_hovernet_args(
                depth,
                n_dense,
                normalization,
                activation,
                convolution,
                attention,
                preactivate,
                preattend,
                use_style,
                merge_policy,
                skip_params,
                upsampling,
            )
            for d in decoders
        }

        # set encoder
        self.encoder = Encoder(
            enc_name,
            depth=depth,
            pretrained=enc_pretrain,
            checkpoint_path=kwargs.get("checkpoint_path", None),
            unettr_kwargs={  # Only used for transformer encoders
                "convolution": convolution,
                "activation": activation,
                "normalization": normalization,
                "attention": attention,
            },
            **encoder_params if encoder_params is not None else {},
        )

        # get the reduction factors for the encoder
        enc_reductions = tuple([inf["reduction"] for inf in self.encoder.feature_info])

        # Style
        self.make_style = None
        if use_style:
            self.make_style = StyleReshape(self.encoder.out_channels[0], style_channels)

        # set decoders and heads
        for decoder_name in decoders:
            decoder = UnetDecoder(
                enc_channels=self.encoder.out_channels,
                enc_reductions=enc_reductions,
                out_channels=out_channels,
                style_channels=style_channels,
                long_skip=long_skip,
                merge_policy=merge_policy,
                n_conv_layers=n_layers,
                n_conv_blocks=n_blocks,
                stage_params=dec_params[decoder_name],
            )
            self.add_module(f"{decoder_name}_decoder", decoder)

        # optional stem skip
        if add_stem_skip:
            for decoder_name in decoders:
                stem_skip = StemSkip(
                    out_channels=out_channels[-1],
                    merge_policy=merge_policy,
                    n_blocks=2,
                    short_skip="residual",
                    block_type="basic",
                    normalization=normalization,
                    activation=activation,
                    convolution=convolution,
                    attention=attention,
                    preactivate=preactivate,
                    preattend=preattend,
                )
                self.add_module(f"{decoder_name}_stem_skip", stem_skip)

        # output heads
        for decoder_name in heads.keys():
            for output_name, n_classes in heads[decoder_name].items():
                seg_head = SegHead(
                    in_channels=decoder.out_channels,
                    out_channels=n_classes,
                    kernel_size=1,
                )
                self.add_module(f"{decoder_name}_{output_name}_seg_head", seg_head)

        self.name = f"HoverNet-{enc_name}"

        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[
            List[torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
        ],
    ]:
        """Forward pass of HoVer-Net.

        Parameters
        ----------
            x : torch.Tensor
                Input image batch. Shape: (B, C, H, W).
            return_feats : bool, default=False
                If True, encoder, decoder, and head outputs will all be returned

        Returns
        -------
        Union[
            Dict[str, torch.Tensor],
            Tuple[
                List[torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
            ],
        ]:
            Dictionary mapping of output names to outputs or if `return_feats == True`
            returns also the encoder features in a list, decoder features as a dict
            mapping decoder names to outputs and the final head outputs dict.
        """
        feats, dec_feats = self.forward_features(x)
        out = self.forward_heads(dec_feats)

        if return_feats:
            return feats, dec_feats, out

        return out


def hovernet_base(type_classes: int, inst_classes: int = 2, **kwargs) -> nn.Module:
    """Create the baseline HoVer-Net (three decoders) from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters
    ----------
        type_classes : int
            Number of type classes.
        inst_classes : int, default=2
            Number of instance classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns
    -------
        nn.Module: The initialized HoVer-Net model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "inst"),
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
            "inst": {"inst": inst_classes},
        },
        **kwargs,
    )

    return hovernet


def hovernet_plus(
    type_classes: int,
    sem_classes: int,
    inst_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Create HoVer-Net+ (additional semantic decoders) from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters
    ----------
        type_classes : int
            Number of type-branch classes.
        sem_classes : int
            Number of semantic-branch classes.
        inst_classes : int, default=2
            Number of instance-branch classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns
    -------
        nn.Module: The initialized HoVer-Net+ model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "inst", "sem"),
        heads={
            "hovernet": {"hovernet": 2},
            "sem": {"sem": sem_classes},
            "type": {"type": type_classes},
            "inst": {"inst": inst_classes},
        },
        **kwargs,
    )

    return hovernet


def hovernet_small(type_classes: int, **kwargs) -> nn.Module:
    """Create HoVer-Net without inst decoder branch from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters
    ----------
        type_classes : int
            Number of type-branch classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns
    -------
        nn.Module: The initialized HoVer-Net-small model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
        },
        **kwargs,
    )

    return hovernet


def hovernet_small_plus(type_classes: int, sem_classes: int, **kwargs) -> nn.Module:
    """Create the HoVer-Net+ without inst decoder branch from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters
    ----------
        type_classes : int
            Number of type-branch classes.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns
    -------
        nn.Module: The initialized HoVer-Net-small+ model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "sem"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return hovernet
