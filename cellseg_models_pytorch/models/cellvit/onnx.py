from pathlib import Path

import torch
from torch import nn
from timm.models.vision_transformer_sam import Attention

from cellseg_models_pytorch.models.base._onnx import _export_onnx
from cellseg_models_pytorch.models.cellvit.cellvit import CellVit

__all__ = ["CellVitONNXWrapper", "export_cellvit_onnx"]


class CellVitONNXWrapper(nn.Module):
    """Return CellVit horizontal/vertical distance maps and class logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction maps and class logits with BCHW axes."""
        nucleus = self.model(x)["nuc"]
        if nucleus.type_map is None:
            raise RuntimeError("ONNX export requires a nuclei cell-type output head.")
        return nucleus.aux_map, nucleus.type_map


def export_cellvit_onnx(
    model: CellVit | nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Save CellVit prediction maps as an ONNX file.

    Args:
        model: A trained ``CellVit`` instance or its PyTorch network.
        output_path: Destination .onnx file. Parent folders are created;
            an existing file may be overwritten.
        input_shape: Example (batch, channels, height, width) shape. Tested with
            ``samvit_base_patch16_224`` at 224 × 224 pixels. Image size stays
            fixed; a dynamic export traces with at least two images.
        opset_version: ONNX operator-set version, default 18.
        dynamic_batch: Allow the number of images in a batch to vary.

    Returns:
        Path to the exported file. Input ``image`` is floating BCHW. Outputs
        ``hv_map`` (horizontal/vertical distance maps) and ``type_map`` (class logits)
        use the same axes. Preprocessing and instance reconstruction stay in Python.

    Raises:
        ValueError: Input dimensions are not four positive integers.
        TypeError: The model is not a CellVit instance or a PyTorch module.
        ImportError: Export requires the optional onnx and onnxscript packages.
        RuntimeError: The PyTorch version is unsupported or export fails.

    Note:
        Export uses the network's device and floating dtype. Every module's
        training mode and SAM attention settings are restored afterward, even
        if export fails.
    """
    network = model.model if isinstance(model, CellVit) else model
    if not isinstance(network, nn.Module):
        raise TypeError("model must be a CellVit instance or torch.nn.Module.")
    attention_states = [
        (module, module.fused_attn)
        for module in network.modules()
        if isinstance(module, Attention)
    ]
    try:
        # ONNX does not support SAM's three-dimensional fused attention inputs.
        for module, _ in attention_states:
            module.fused_attn = False
        return _export_onnx(
            network,
            CellVitONNXWrapper,
            ("hv_map", "type_map"),
            output_path,
            input_shape,
            opset_version,
            dynamic_batch,
        )
    finally:
        for module, fused_attn in attention_states:
            module.fused_attn = fused_attn
