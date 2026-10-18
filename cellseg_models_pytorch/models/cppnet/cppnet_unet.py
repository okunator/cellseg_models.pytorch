from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.decoders.multitask_decoder import (
    DecoderSoftOutput,
    MultiTaskDecoder,
)
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.base._seg_head import SegHead
from cellseg_models_pytorch.models.cppnet._conf import _create_cppnet_args
from cellseg_models_pytorch.models.cppnet.sampling import SamplingFeatures

__all__ = [
    "CPPNetUnet",
    "cppnet_nuclei",
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

        Args:
            stardist_map: The stardist map. Shape: (B, n_rays, H, W)
            features: The features from the encoder. Shape: (B, C, H, W)

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


class CPPNetUnet(nn.ModuleDict):
    """A CPPNet family network with configurable output branches.

    Use the family factory for standard nucleus heads. Direct construction is
    for choosing your own branches and head channel counts. ``enc_name`` names
    the encoder stored at ``network[network.enc_name]``.
    """

    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        n_rays: int = 32,
        erosion_factors: Tuple[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        depth: int = 4,
        out_channels: Tuple[int, ...] = (256, 128, 64, 32),
        style_channels: int = None,
        enc_name: str = "efficientnet_b5",
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
        """Configure the CPPNetUnet network and its output heads.

        CPP-Net:
        - CPP-Net: Context-aware Polygon Proposal Network for Nucleus Segmentation
        - https://arxiv.org/abs/2102.06867

        Args:
            decoders: Names of decoder branches. Each name must have an entry in heads.
            heads: Output names and channel counts grouped by decoder. Use
                ``nuc_stardist`` for the auxiliary head and ``nuc_type`` for
                cell-type logits; ray-based reconstruction also needs ``nuc_binary``.
            n_rays: Number of radial distance channels; match the configured ray head.
            erosion_factors: Sampling distances used when refining the initial ray predictions.
            depth: Number of decoder stages; align stage settings and encoder features.
            out_channels: Channel counts at each decoder stage, from deepest to shallowest.
            style_channels: Size of the vector describing image appearance. None disables its use.
            enc_name: timm encoder name. Access it through network[network.enc_name].
            enc_pretrain: Load pretrained encoder weights; may download on first use.
            enc_freeze: Disable encoder weight gradients; does not keep it in evaluation mode.
            enc_out_indices: Encoder feature indices. None selects range(depth).
            upsampling: Method for increasing feature resolution between decoder stages.
            long_skip: Connections from encoder features to decoder stages, such as "unet".
            merge_policy: Combine connected features by "sum" or "cat" (concatenation).
            short_skip: Connections within decoder blocks: "basic", "residual" or "dense".
            block_type: Convolution block type used within each decoder stage.
            normalization: Feature normalization, such as "bn" (batch normalization), or None.
            activation: Activation function used within decoder blocks, such as "relu".
            convolution: Convolution implementation: "conv", "wsconv" or "scaled_wsconv".
            preactivate: Apply normalization and activation before the convolution.
            attention: Channel attention method, or None to disable it.
            preattend: Apply configured attention before the convolution block.
            out_size: Optional square output size in pixels. None keeps the decoder output size.
            encoder_kws: Additional keyword arguments passed to the timm encoder.
            skip_kws: Additional settings for the selected skip-connection implementation.
            stem_skip_kws: Additional settings for the connection from the input image to the decoder.
            inst_key: Output kind used to identify the instance foreground head.
            **kwargs: Additional keyword arguments are accepted but unused.
        """
        super().__init__()
        self.out_size = out_size
        self.inst_key = inst_key
        self.aux_key = "stardist"
        self.n_rays = n_rays
        self.enc_name = enc_name

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
        self.add_module(
            self.enc_name,
            Encoder(
                timm_encoder_name=enc_name,
                timm_encoder_out_indices=enc_out_indices,
                timm_encoder_pretrained=enc_pretrain,
                timm_extra_kwargs=encoder_kws,
            ),
        )

        self.decoder = MultiTaskDecoder(
            decoders=decoders,
            heads=heads,
            out_channels=out_channels,
            enc_feature_info=self[self.enc_name].feature_info,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stage_kws=stage_kws,
            stem_skip_kws=stem_skip_kws,
            long_skip=long_skip,
            out_size=out_size,
            style_channels=style_channels,
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
            self[self.enc_name].freeze_encoder()

    def forward(self, x: torch.Tensor, return_pred_only: bool = True) -> Dict[str, Any]:
        """Compute raw CPPNet prediction maps for training or inference.

        Args:
            x: Floating-point RGB batch with shape (B, 3, H, W), on the network's
                device and dtype. B is batch size, H height and W width. Apply the
                normalization used during training before calling the network.
            return_pred_only: False also includes encoder/decoder features and
                encoder output under ``enc_feats``, ``dec_feats`` and ``enc_out``.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` entries; unused outputs
            are None. Instance outputs contain raw class scores (logits) in
            ``type_map``, maps for separating objects in ``aux_map``, and optionally
            foreground values in ``binary_map``. Map shape is (B, channels, H, W),
            using the configured output size. Maps keep the network device, dtype
            and gradients for training. Features are included only when requested.
        """
        enc_output, feats = self[self.enc_name](x)
        dec_out: DecoderSoftOutput = self.decoder(feats, x)
        if dec_out.nuc_map is not None:
            dec_name = dec_out.nuc_map.parents["aux_map"][0]
            ray_refined, _ = self.cppnet_refine(
                dec_out.nuc_map.aux_map, dec_out.dec_feats[dec_name][-1]
            )
            dec_out.nuc_map.aux_map = ray_refined

        if dec_out.cyto_map is not None:
            dec_name = dec_out.cyto_map.parents["aux_map"][0]
            ray_refined, _ = self.cppnet_refine(
                dec_out.cyto_map.aux_map, dec_out.dec_feats[dec_name][-1]
            )
            dec_out.nuc_map.aux_map = ray_refined

        res = {
            "nuc": dec_out.nuc_map,
            "tissue": dec_out.tissue_map,
            "cyto": dec_out.cyto_map,
        }

        if not return_pred_only:
            res["enc_feats"] = dec_out.enc_feats
            res["dec_feats"] = dec_out.dec_feats
            res["enc_out"] = enc_output

        return res


def cppnet_nuclei(n_rays: int, n_nuc_classes: int, **kwargs) -> nn.Module:
    """Create a CPPNet network with standard nucleus heads.

    - https://arxiv.org/abs/2102.06867

    Args:
        n_rays: Number of radial distances per pixel; match the training targets.
        n_nuc_classes: Number of cell-type output channels, including any background.
        **kwargs: Network options accepted by ``CPPNetUnet``, such as
            ``enc_name`` and ``enc_pretrain=False`` for offline construction.
            Decoder branches and head definitions are set by this factory.

    Returns:
        A ``CPPNetUnet`` PyTorch module. Call it with a normalized RGB batch
        to obtain raw output maps; it does not reconstruct individual instances.
    """
    cppnet = CPPNetUnet(
        decoders=("stardist", "type"),
        heads={
            "stardist": {"nuc_stardist": n_rays, "nuc_binary": 1},
            "type": {"nuc_type": n_nuc_classes},
        },
        n_rays=n_rays,
        **kwargs,
    )

    return cppnet
