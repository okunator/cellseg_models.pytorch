from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import MultiTaskDecoder
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.stardist._conf import _create_stardist_args

__all__ = ["StarDistUnet", "stardist_base", "stardist_base_multiclass", "stardist_plus"]


class StarDistUnet(nn.Module):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_indices: Tuple[int, ...] = None,
        upsampling: str = "fixed-unpool",
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
        out_size: int = None,
        encoder_kws: Dict[str, Any] = None,
        skip_kws: Dict[str, Any] = None,
        stem_skip_kws: Dict[str, Any] = None,
        inst_key: str = "dist",
        **kwargs,
    ) -> None:
        """Stardist (2D) U-Net model implementation.

        Stardist:
            - https://arxiv.org/abs/1806.03535

        Note:
            Minor differences from the original implementation.
            - long skip concatenation/sum applied before each conv layer rather than after.

        Parameters:
            decoders (Tuple[str, ...]):
                Names of the decoder branches of this network. E.g. ("stardist", "sem")
            heads (Dict[str, Dict[str, int]]):
                The decoder branches mapped to segmentation heads E.g.
                {"stardist": {"type": 4, "stardist": 32}, "sem": {"sem": 5}}
            depth (int, default=4):
                The depth of the encoder. I.e. Number of returned feature maps from
                the encoder. Maximum depth = 5.
            out_channels (Tuple[int, ...], default=(256, 128, 64, 32)):
                Out channels for each decoder stage.
            style_channels (int, default=256):
                Number of style vector channels. If None, style vectors are ignored.
            enc_name (str, default="resnet50"):
                Name of the encoder. See timm docs for more info.
            enc_pretrain (bool, default=True):
                Whether to use imagenet pretrained weights in the encoder.
            enc_freeze (bool, default=False):
                Freeze encoder weights for training.
            enc_out_indices (Tuple[int, ...], default=None):
                Indices of the encoder output features. If None, indices is set to
                `range(len(depth))`.
            upsampling (str, default="fixed-unpool"):
                The upsampling method. One of: "fixed-unpool", "nearest", "bilinear",
                "bicubic", "conv_transpose"
            long_skip (str, default="unet"):
                long skip method to be used. One of: "unet", "unetpp", "unet3p",
                "unet3p-lite", None
            merge_policy (str, default="sum"):
                The long skip merge policy. One of: "sum", "cat"
            short_skip (str, default="basic"):
                The name of the short skip method. One of: "residual", "dense", "basic"
            block_type (str, default="basic"):
                The type of the convolution block type. One of: "basic". "mbconv",
                "fmbconv" "dws", "bottleneck".
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
            out_size (int, optional):
                If specified, the output size of the model will be (out_size, out_size).
                I.e. the outputs will be interpolated to this size.
            encoder_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the encoder. See timm docs for more info.
            skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the skip-connection module.
            stem_skip_kws (Dict[str, Any], default=None):
                Extra keyword arguments for the stem skip-connection module.
            inst_key (str, default="dist"):
                The key for the model output that will be used in the instance
                segmentation post-processing pipeline as the binary segmentation result.
        """
        super().__init__()
        self.out_size = out_size
        self.inst_key = inst_key
        self.aux_key = "stardist"

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        # create decoder args
        n_layers = (1,) * depth
        n_blocks = ((2,),) * depth
        stage_kws = _create_stardist_args(
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
            head_excitation_channels=128,
        )

        self.name = f"StardistUnet-{enc_name}"

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
        """Forward pass of Stardist.

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


def stardist_base(n_rays: int, **kwargs) -> nn.Module:
    """Create the Stardist U-net from kwargs.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns:
        nn.Module: The initialized Stardist U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist",),
        heads={"stardist": {"stardist": n_rays, "dist": 1}},
        **kwargs,
    )

    return stardist_unet


def stardist_base_multiclass(n_rays: int, n_type_classes: int, **kwargs) -> nn.Module:
    """Create the Stardist U-net with an extra multi-class segmentation head.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters:
        n_rays (int):
            Number of rays predicted per each object.
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns:
        nn.Module: The initialized multiclass Stardist U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist",),
        heads={"stardist": {"stardist": n_rays, "dist": 1, "type": n_type_classes}},
        **kwargs,
    )

    return stardist_unet


def stardist_plus(
    n_rays: int, n_type_classes: int, n_sem_classes: int, **kwargs
) -> nn.Module:
    """Create the Stardist U-net with a semantic decoder-branch.

    Stardist:
    - https://arxiv.org/abs/1806.03535

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the StarDistUnet class.

    Returns:
        nn.Module: The initialized Stardist+ U-net model.
    """
    stardist_unet = StarDistUnet(
        decoders=("stardist", "sem"),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1, "type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        **kwargs,
    )

    return stardist_unet
