from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders import Decoder
from cellseg_models_pytorch.decoders.long_skips import StemSkip
from cellseg_models_pytorch.encoders import EncoderUnetTR
from cellseg_models_pytorch.encoders.vit_det_SAM import build_sam_encoder
from cellseg_models_pytorch.models.base._base_model import BaseMultiTaskSegModel
from cellseg_models_pytorch.models.base._seg_head import SegHead
from cellseg_models_pytorch.modules.misc_modules import StyleReshape

from ._conf import _create_cellvit_args

__all__ = [
    "CellVitSAM",
    "cellvit_sam_base",
    "cellvit_sam_small",
    "cellvit_sam_plus",
    "cellvit_sam_small_plus",
]


class CellVitSAM(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        inst_key: str = "inst",
        out_channels: Tuple[int, ...] = (512, 256, 128, 64),
        encoder_out_channels: Tuple[int, ...] = (512, 512, 256, 128),
        layer_depths: Tuple[int, ...] = (3, 2, 2, 2),
        style_channels: int = None,
        enc_name: str = "sam_vit_b",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        long_skip: str = "unet",
        merge_policy: str = "cat",
        short_skip: str = "basic",
        normalization: str = "bn",
        activation: str = "relu",
        convolution: str = "conv",
        preactivate: bool = True,
        attention: str = None,
        preattend: bool = False,
        add_stem_skip: bool = True,
        skip_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Create a CellVit model.

        CellVit:
        - https://arxiv.org/abs/2306.15350

                          (|------ SEMANTIC_DECODER ----- SEMANTIC_HEAD)
                           |
                           |------ TYPE_DECODER ---------- TYPE_HEAD
        UnetTR-SAM-encoder-|
                           |------ HOVER_DECODER --------- HOVER_HEAD
                           |
                          (|------ INSTANCE_DECODER ---- INSTANCE_HEAD)

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
            encoder_out_channels : Tuple[int, ...], default=(512, 512, 256, 128)
                Out channels for each SAM-UnetTR encoder stage.
            out_channels : Tuple[int, ...], default=(256, 256, 64, 64)
                Out channels for each decoder stage.
            layer_depths : Tuple[int, ...], default=(4, 4, 4, 4)
                The number of conv blocks at each decoder stage.
            style_channels : int, default=None
                Number of style vector channels. If None, style vectors are ignored.
            enc_name : str, default="sam_vit_b"
                Name of the encoder. One of: "sam_vit_b", "sam_vit_l", "sam_vit_h",
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
            add_stem_skip : bool, default=True
                If True, a stem conv block is added to the model whose output is used
                as a long skip input at the final decoder layer that is the highest
                resolution layer and the same resolution as the input image.
            skip_params : Optional[Dict]
                Extra keyword arguments for the skip-connection modules. These depend
                on the skip module. Refer to specific skip modules for more info. I.e.
                `UnetSkip`, `UnetppSkip`, `Unet3pSkip`.

        Raises
        ------
            ValueError: If `decoders` does not contain 'hovernet'.
            ValueError: If `heads` keys don't match `decoders`.
            ValueError: If decoder names don't have a matching head name in `heads`.
        """
        super().__init__()
        self.aux_key = self._check_decoder_args(decoders, ("hovernet",))
        self.inst_key = inst_key
        self.depth = 4
        self._check_head_args(heads, decoders)
        self._check_depth(self.depth, {"out_channels": out_channels})

        self.add_stem_skip = add_stem_skip
        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create decoder build args
        n_layers = (1,) * self.depth
        n_blocks = tuple([(d,) for d in layer_depths])

        dec_params = {
            d: _create_cellvit_args(
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

        if enc_name not in ("sam_vit_b", "sam_vit_l", "sam_vit_h"):
            raise ValueError(
                f"Wrong encoder name. Got: {enc_name}. "
                "Allowed encoder for CellVit: sam_vit_b, sam_vit_l, sam_vit_h."
            )

        # set encoder
        self.encoder = EncoderUnetTR(
            backbone=build_sam_encoder(name=enc_name, pretrained=enc_pretrain),
            out_channels=encoder_out_channels,
            up_method="conv_transpose",
            convolution=convolution,
            activation=activation,
            normalization=normalization,
            attention=attention,
        )

        # get the reduction factors for the encoder
        enc_reductions = tuple([inf["reduction"] for inf in self.encoder.feature_info])

        # Style
        self.make_style = None
        if use_style:
            self.make_style = StyleReshape(self.encoder.out_channels[0], style_channels)

        # set decoders and heads
        for decoder_name in decoders:
            decoder = Decoder(
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
                    short_skip="basic",
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
                self.add_module(f"{output_name}_seg_head", seg_head)

        self.name = f"CellVit-{enc_name}"

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
        """Forward pass of CellVit-SAM.

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


def cellvit_sam_base(
    enc_name: str, type_classes: int, inst_classes: int = 2, **kwargs
) -> nn.Module:
    """Create the baseline CellVit-SAM (three decoders) from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters
    ----------
        enc_name : str
            Name of the encoder. One of: "sam_vit_b", "sam_vit_l", "sam_vit_h",
        type_classes : int
            Number of type classes.
        inst_classes : int, default=2
            Number of instance classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns
    -------
        nn.Module: The initialized CellVitSAM model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "inst"),
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
            "inst": {"inst": inst_classes},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_plus(
    enc_name: str,
    type_classes: int,
    sem_classes: int,
    inst_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Create CellVit-SAM (additional semantic decoders) from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters
    ----------
        enc_name : str
            Name of the encoder. One of: "sam_vit_b", "sam_vit_l", "sam_vit_h",
        type_classes : int
            Number of type-branch classes.
        sem_classes : int
            Number of semantic-branch classes.
        inst_classes : int, default=2
            Number of instance-branch classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns
    -------
        nn.Module: The initialized CellVitSAM+ model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "inst", "sem"),
        heads={
            "hovernet": {"hovernet": 2},
            "sem": {"sem": sem_classes},
            "type": {"type": type_classes},
            "inst": {"inst": inst_classes},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_small(enc_name: str, type_classes: int, **kwargs) -> nn.Module:
    """Create CellVit-SAM without inst decoder branch from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters
    ----------
        enc_name : str
            Name of the encoder. One of: "sam_vit_b", "sam_vit_l", "sam_vit_h",
        type_classes : int
            Number of type-branch classes.
        **kwargs:
            Arbitrary key word args for the CellVitSAM class.

    Returns
    -------
        nn.Module: The initialized CellVitSAM-small model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
        },
        **kwargs,
    )

    return cellvit_sam


def cellvit_sam_small_plus(
    enc_name: str, type_classes: int, sem_classes: int, **kwargs
) -> nn.Module:
    """Create the CellVit-SAM+ without inst decoder branch from kwargs.

    CellVit:
        - https://arxiv.org/abs/2306.15350

    Parameters
    ----------
        enc_name : str
            Name of the encoder. One of: "sam_vit_b", "sam_vit_l", "sam_vit_h",
        type_classes : int
            Number of type-branch classes.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the CellVitsSAM class.

    Returns
    -------
        nn.Module: The initialized CellVit-SAM-small+ model.
    """
    cellvit_sam = CellVitSAM(
        enc_name=enc_name,
        decoders=("hovernet", "type", "sem"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return cellvit_sam
