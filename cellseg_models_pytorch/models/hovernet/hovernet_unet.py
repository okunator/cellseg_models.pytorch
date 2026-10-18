from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from cellseg_models_pytorch.decoders.multitask_decoder import (
    DecoderSoftOutput,
    MultiTaskDecoder,
)
from cellseg_models_pytorch.encoders import Encoder
from cellseg_models_pytorch.models.hovernet._conf import _create_hovernet_args

__all__ = [
    "HoverNetUnet",
    "hovernet_nuclei",
]


class HoverNetUnet(nn.ModuleDict):
    """A HoverNet family network with configurable output branches.

    Use the family factory for standard nucleus heads. Direct construction is
    for choosing your own branches and head channel counts. ``enc_name`` names
    the encoder stored at ``network[network.enc_name]``.
    """

    def __init__(
        self,
        decoders: Tuple[str, ...],
        heads: Dict[str, Dict[str, int]],
        depth: int = 4,
        out_channels: Tuple[int, ...] = (512, 256, 64, 64),
        style_channels: int = None,
        enc_name: str = "efficientnet_b5",
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
        """Configure the HoverNetUnet network and its output heads.

        Based on `HoVer-Net <https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub>`_.
        This version supports timm encoders and uses transition convolutions
        between dense blocks.

        Args:
            decoders: Names of decoder branches. Each name must have an entry in heads.
            heads: Output names and channel counts grouped by decoder. Use
                ``nuc_hovernet`` for the auxiliary head and ``nuc_type`` for
                cell-type logits.
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
            n_dense: Number of densely connected blocks at each decoder stage.
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
        self.inst_key = inst_key
        self.aux_key = "hovernet"
        self.enc_name = enc_name

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

        # init decoder weights
        self.decoder.initialize()

        # freeze encoder if specified
        if enc_freeze:
            self[self.enc_name].freeze_encoder()

        self.name = f"HoverNet-{enc_name}"

    def forward(self, x: torch.Tensor, return_pred_only: bool = True) -> Dict[str, Any]:
        """Compute raw HoverNet prediction maps for training or inference.

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


def hovernet_nuclei(n_nuc_classes: int, **kwargs) -> nn.Module:
    """Create a HoverNet network with standard nucleus heads.

    - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

    Args:
        n_nuc_classes: Number of cell-type output channels, including any background.
        **kwargs: Network options accepted by ``HoverNetUnet``, such as
            ``enc_name`` and ``enc_pretrain=False`` for offline construction.
            Decoder branches and head definitions are set by this factory.

    Returns:
        A ``HoverNetUnet`` PyTorch module. Call it with a normalized RGB batch
        to obtain raw output maps; it does not reconstruct individual instances.
    """
    hovernet = HoverNetUnet(
        decoders=("hovernet", "type"),
        heads={
            "hovernet": {"nuc_hovernet": 2},
            "type": {"nuc_type": n_nuc_classes},
        },
        **kwargs,
    )

    return hovernet
