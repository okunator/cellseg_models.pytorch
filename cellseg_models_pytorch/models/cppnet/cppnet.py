from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.decoders import UnetDecoder
from cellseg_models_pytorch.decoders.long_skips import StemSkip
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.modules.misc_modules import StyleReshape

from ..base._base_model import BaseMultiTaskSegModel
from ..base._seg_head import SegHead
from ._conf import _create_cppnet_args
from .sampling import SamplingFeatures

__all__ = [
    "CPPNet",
    "cppnet_base",
    "cppnet_base_multiclass",
    "cppnet_plus",
]


class CPPNet(BaseMultiTaskSegModel):
    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        erosion_factors: Tuple[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        inst_key: str = "dist",
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "resnet50",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
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
        add_stem_skip: bool = False,
        out_size: Optional[int] = None,
        skip_params: Optional[Dict] = None,
        encoder_params: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Create CPP-Net.

        CPP-Net:
        - CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation
        - https://arxiv.org/abs/2102.06867

                (|------ SEMANTIC_DECODER ----- SEMANTIC_HEAD)
                 |
                 |------ TYPE_DECODER ---------- TYPE_HEAD
        ENCODER -|
                 |                    |-- RAY_HEAD --------- REFINING ------- +
                 |- STARDIST_DECODER -|                                       |
                                      |-- CONFIDENCE_HEAD0 - REFINING - CONFIDENCE_HEAD1
                                      |
                                      |-- DISTPROB_HEAD


        Parameters
        ----------
            decoders : Tuple[str, ...]
                Names of the decoder branches of this network. E.g. ("cppnet", "sem")
            heads : Dict[str, Dict[str, int]]
                The segmentation heads of the architecture. I.e. Names of the decoder
                branches (has to match `decoders`) mapped to dicts
                of output name - number of output classes. E.g.
                {"cppnet": {"cppnet": 2}, "sem": {"sem": 5}, "type": {"type": 5}}
            inst_key : str, default="dist"
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
            short_skip : str, default="basic"
                The name of the short skip method. One of: "residual", "dense", "basic"
            block_type : str, default="basic"
                The type of the convolution block type. One of: "basic". "mbconv",
                "fmbconv" "dws", "bottleneck".
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
        self._check_decoder_args(decoders, ("stardist",))
        self.aux_key = "stardist_refined"
        self.inst_key = inst_key
        self._check_head_args(heads, decoders)
        self._check_depth(depth, {"out_channels": out_channels})

        self.add_stem_skip = add_stem_skip
        self.enc_freeze = enc_freeze
        use_style = style_channels is not None
        self.heads = heads

        n_layers = (1,) * depth
        n_blocks = ((2,),) * depth
        dec_params = {
            d: _create_cppnet_args(
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
                upsampling,
            )
            for d in decoders
        }

        # set encoder
        # self.encoder = Encoder(
        #     enc_name,
        #     depth=depth,
        #     pretrained=enc_pretrain,
        #     checkpoint_path=kwargs.get("checkpoint_path", None),
        #     unettr_kwargs={  # Only used for transformer encoders
        #         "convolution": convolution,
        #         "activation": activation,
        #         "normalization": normalization,
        #         "attention": attention,
        #     },
        #     **encoder_params if encoder_params is not None else {},
        # )
        self.encoder = Encoder(
            timm_encoder_name=enc_name,
            timm_encoder_out_indices=tuple(range(depth)),
            pixel_decoder_out_channels=out_channels,
            timm_encoder_pretrained=enc_pretrain,
            timm_extra_kwargs=encoder_params,
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

        # add cppnet specific modules
        self.erosion_factors = list(erosion_factors)
        n_rays = self.heads["stardist"]["stardist"]
        self.heads["stardist"]["confidence"] = 1
        self.heads["stardist"]["stardist_refined"] = n_rays
        self.conv_0_confidence = SegHead(
            decoder.out_channels, n_rays, kernel_size=1, bias=False
        )
        self.conv_1_confidence = SegHead(
            1 + len(erosion_factors), 1 + len(erosion_factors), bias=True
        )
        nn.init.constant_(self.conv_1_confidence.head.bias, 1.0)
        self.sampling_features = SamplingFeatures(n_rays=n_rays)
        self.final_activation_ray = nn.ReLU(inplace=True)

        # set model name
        self.name = f"CPPNet-{enc_name}"

        # init decoder weights
        self.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self.freeze_encoder()

    def cppnet_refine(
        self, stardist_map: torch.Tensor, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine the stardist map and confidence map.

        Parameters
        ----------
            stardist_map : torch.Tensor
                The stardist map. Shape: (B, n_rays, H, W)
            features : torch.Tensor
                The features from the encoder. Shape: (B, C, H, W)

        Returns
        -------
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

    def forward(
        self,
        x: torch.Tensor,
        return_feats: bool = False,
        return_confidence: bool = False,
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[
            List[torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
        ],
    ]:
        """Forward pass of CPP-Net.

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
        _, feats, dec_feats = self.forward_features(x)
        out = self.forward_heads(dec_feats)

        # cppnet specific
        ray_refined, confidence_refined = self.cppnet_refine(
            out["stardist"], dec_feats["stardist"][-1]
        )
        out["stardist_refined"] = ray_refined

        if return_confidence:
            out["confidence"] = confidence_refined

        if return_feats:
            return feats, dec_feats, out

        return out


def cppnet_base(n_rays: int, **kwargs) -> nn.Module:
    """Create the baseline CPP-Net from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns
    -------
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist",),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
        },
        **kwargs,
    )

    return cppnet


def cppnet_base_multiclass(n_rays: int, type_classes: int, **kwargs) -> nn.Module:
    """Create the baseline CPP-Net with a type classification branch from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        type_classes : int
            Number of type classes.
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns
    -------
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist", "type"),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": type_classes},
        },
        **kwargs,
    )

    return cppnet


def cppnet_plus(
    n_rays: int, type_classes: int, sem_classes: int, **kwargs
) -> nn.Module:
    """Create the CPP-Net with a type and semantic classification branch from kwargs.

    CPP-Net:
        - https://arxiv.org/abs/2102.06867

    Parameters
    ----------
        n_rays : int
            Number of rays predicted per each object
        type_classes : int
            Number of type classes.
        sem_classes : int
            Number of semantic-branch classes.
        **kwargs:
            Arbitrary key word args for the CPPNet class.

    Returns
    -------
        nn.Module: The initialized CPP-Net model.
    """
    cppnet = CPPNet(
        decoders=("stardist", "type", "sem"),
        heads={
            "stardist": {"stardist": n_rays, "dist": 1},
            "type": {"type": type_classes},
            "sem": {"sem": sem_classes},
        },
        **kwargs,
    )

    return cppnet
