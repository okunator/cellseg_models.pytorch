from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.decoders.multitask_decoder import MultiTaskDecoder
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.base._seg_head import SegHead
from cellseg_models_pytorch.models.cppnet._conf import _create_cppnet_args
from cellseg_models_pytorch.models.cppnet.sampling import SamplingFeatures

__all__ = [
    "CPPNet",
    "cppnet_base",
    "cppnet_base_multiclass",
    "cppnet_plus",
]


class CPPRefine(nn.Module):
    def __init__(
        self, in_channels: int, erosion_factors: Tuple[float, ...], n_rays: int
    ) -> None:
        """CPP-Net ray map refining module."""
        super().__init__()
        self.erosion_factors = list(erosion_factors)
        self.conv_0_confidence = SegHead(in_channels, n_rays, kernel_size=1, bias=False)
        self.conv_1_confidence = SegHead(
            1 + len(erosion_factors), 1 + len(erosion_factors), bias=True
        )
        nn.init.constant_(self.conv_1_confidence.head.bias, 1.0)
        self.sampling_features = SamplingFeatures(n_rays=n_rays)
        self.final_activation_ray = nn.ReLU(inplace=True)

    def forward(
        self, stardist_map: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine the stardist map and confidence map.

        Parameters:
            stardist_map (torch.Tensor):
                The stardist map. Shape: (B, n_rays, H, W)
            features (torch.Tensor):
                The features from the encoder. Shape: (B, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                - refined stardist map. Shape: (B, n_rays, H, W)
                - refined confidence map. Shape: (B, C, H, W)
        """
        # cppnet specific ops
        out_confidence = self.conv_0_confidence(features)
        out_ray_for_sampling = stardist_map

        ray_refined = [out_ray_for_sampling]
        confidence_refined = [out_confidence]

        for erosion_factor in self.erosion_factors:
            base_dist = (out_ray_for_sampling - 1.0) * erosion_factor
            ray_sampled, _, _ = self.sampling_features(
                out_ray_for_sampling, base_dist, 1
            )
            conf_sampled, _, _ = self.sampling_features(out_confidence, base_dist, 1)
            ray_refined.append(ray_sampled + base_dist)
            confidence_refined.append(conf_sampled)
        ray_refined = torch.stack(ray_refined, dim=1)
        b, k, c, h, w = ray_refined.shape

        confidence_refined = torch.stack(confidence_refined, dim=1)
        confidence_refined = (
            confidence_refined.permute([0, 2, 1, 3, 4])
            .contiguous()
            .view(b * c, k, h, w)
        )
        confidence_refined = self.conv_1_confidence(confidence_refined)
        confidence_refined = confidence_refined.view(b, c, k, h, w).permute(
            [0, 2, 1, 3, 4]
        )
        confidence_refined = F.softmax(confidence_refined, dim=1)

        ray_refined = (ray_refined * confidence_refined).sum(dim=1)
        ray_refined = self.final_activation_ray(ray_refined)

        return ray_refined, confidence_refined


class CPPNet(nn.Module):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        n_rays: int,
        erosion_factors: Tuple[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        enc_out_indices: Tuple[int, ...] = None,
        upsampling: str = "conv_transpose",
        long_skip: str = "unet",
        merge_policy: str = "cat",
        short_skip: str = "basic",
        block_type: str = "basic",
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
        inst_key: str = "dist",
        **kwargs,
    ) -> None:
        """Create CPP-Net.

        CPP-Net:
        - CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation
        - https://arxiv.org/abs/2102.06867

        Parameters:
            decoders (Tuple[str, ...]):
                Names of the decoder branches of this network. E.g. ("cppnet", "sem")
            heads (Dict[str, Dict[str, int]]):
                The decoder branches mapped to segmentation heads E.g.
                {"cppnet": {"type": 4, "stardist": 32}, "sem": {"sem": 5}}
            n_rays (int):
                Number of rays predicted per object.
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
        self.n_rays = n_rays

        if enc_out_indices is None:
            enc_out_indices = tuple(range(depth))

        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        n_layers = (1,) * depth
        n_blocks = ((2,),) * depth
        stage_kws = _create_cppnet_args(
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
            # head_excitation_channels=128,
        )

        self.cppnet_refine = CPPRefine(
            in_channels=out_channels[-1],
            erosion_factors=erosion_factors,
            n_rays=self.n_rays,
        )

        self.name = f"CPP-Net-{enc_name}"

        # init decoder weights
        self.decoder.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.encoder.freeze_encoder()

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
        return_confidence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of CPP-Net.

        Parameters:
            x (torch.Tensor):
                Input image batch. Shape: (B, C, H, W).
            return_feats (bool, default=False):
                If True, encoder, decoder, and head outputs will all be returned
            return_confidence (bool, default=False):
                If True, confidence map will be returned

        Returns:
            Dict[str, torch.Tensor]:
                Dictionary of outputs. if `return_feats == True` returns also the encoder
                output, a list of encoder features, dict of decoder features and the head
                outputs (segmentations) dict.
        """
        enc_output, feats = self.encoder.forward(x)
        feats, dec_feats, out = self.decoder.forward(feats, x)

        # cppnet specific operations
        for key in out.keys():
            dec_name, head_name = key.split("-")
            if self.aux_key in head_name:
                ray_refined, confidence_refined = self.cppnet_refine(
                    out[key], dec_feats[dec_name][-1]
                )
                out[key] = ray_refined

            if return_confidence:
                out[f"{key}_confidence"] = confidence_refined

        if return_feats:
            return enc_output, feats, dec_feats, out

        return out


def cppnet_base(n_rays: int, **kwargs) -> nn.Module:
    """Create the baseline CPP-Net from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist",),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
        },
        n_rays=n_rays,
        **kwargs,
    )

    return cppnet


def cppnet_base_multiclass(n_rays: int, n_type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline CPP-Net with a type classification branch from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        n_type_classes (int):
            Number of cell type classes.
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist", "type"),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": n_type_classes},
        },
        n_rays=n_rays,
        **kwargs,
    )

    return cppnet


def cppnet_plus(
    n_rays: int, n_type_classes: int, n_sem_classes: int, **kwargs
) -> nn.Module:
    """Create the CPP-Net with a type and semantic classification branch from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters:
        n_rays (int):
            Number of rays predicted per each object
        n_type_classes (int):
            Number of cell type classes.
        n_sem_classes (int):
            Number of tissue type classes.
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns:
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist", "type", "sem"),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": n_type_classes},
            "sem": {"sem": n_sem_classes},
        },
        n_rays=n_rays,
        **kwargs,
    )

    return cppnet
