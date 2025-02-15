from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import MultiTaskDecoder
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.hovernet._conf import _create_hovernet_args

__all__ = [
    "HoverNet",
    "hovernet_base",
    "hovernet_small",
    "hovernet_plus",
    "hovernet_small_plus",
]


class HoverNet(nn.Module):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        depth: int = 4,
        out_channels: Tuple[int, ...] = (512, 256, 64, 64),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_indices: Tuple[int, ...] = None,
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
        out_size: int = None,
        encoder_kws: Dict[str, Any] = None,
        skip_kws: Dict[str, Any] = None,
        stem_skip_kws: Dict[str, Any] = None,
        inst_key: str = "inst",
        **kwargs,
    ) -> None:
        """Hover-Net implementation.

        HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

        Note:
            Minor differences from the original implementation.
            - Different encoder, (any encoder from timm-library).
            - Dense blocks have transition conv-blocks like in the original dense-net.

        Parameters:
            decoders : Tuple[str, ...]
                Names of the decoder branches of this network. E.g. ("hovernet", "sem")
            heads : Dict[str, Dict[str, int]]
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                of output name - number of output classes. E.g.
                {"hovernet": {"hovernet": 2}, "sem": {"sem": 5}, "type": {"type": 5}}
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

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # Create decoder build args
        n_layers = (3, 3) + (1,) * (depth - 2)
        n_blocks = ((1, n_dense[0], 1), (1, n_dense[1], 1)) + ((1,),) * (depth - 2)
        stage_kws = _create_hovernet_args(
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

        self.name = f"HoverNet-{enc_name}"

        # init decoder weights
        self.decoder.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.encoder.freeze_encoder()

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
        dec_feats, out = self.decoder.forward(feats, x)

        if return_feats:
            return enc_output, feats, dec_feats, out

        return out


def hovernet_base(n_type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline HoVer-Net (three decoders) from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters:
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns:
        nn.Module: The initialized HoVer-Net model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "inst"),
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
            "inst": {"inst": 2},
        },
        **kwargs,
    )

    return hovernet


def hovernet_plus(
    n_type_classes: int,
    n_sem_classes: int,
    **kwargs,
) -> nn.Module:
    """Create HoVer-Net+ (additional semantic decoders) from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters:
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns:
        nn.Module: The initialized HoVer-Net+ model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "inst", "sem"),
        heads={
            "hovernet": {"hovernet": 2},
            "sem": {"sem": n_sem_classes},
            "type": {"type": n_type_classes},
            "inst": {"inst": 2},
        },
        **kwargs,
    )

    return hovernet


def hovernet_small(n_type_classes: int, **kwargs) -> nn.Module:
    """Create HoVer-Net without inst decoder branch from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters:
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns:
        nn.Module: The initialized HoVer-Net-small model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
        },
        **kwargs,
    )

    return hovernet


def hovernet_small_plus(n_type_classes: int, n_sem_classes: int, **kwargs) -> nn.Module:
    """Create the HoVer-Net+ without inst decoder branch from kwargs.

    HoVer-Net:
        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Parameters
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the HoverNet class.

    Returns:
        nn.Module: The initialized HoVer-Net-small+ model.
    """
    hovernet = HoverNet(
        decoders=("hovernet", "type", "sem"),
        inst_key="type",
        heads={
            "hovernet": {"hovernet": 2},
            "type": {"type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        **kwargs,
    )

    return hovernet
