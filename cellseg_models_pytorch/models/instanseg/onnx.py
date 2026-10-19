from pathlib import Path

import torch
from torch import nn

from cellseg_models_pytorch.models.base._onnx import _export_onnx
from cellseg_models_pytorch.models.instanseg.instanseg import InstanSeg

__all__ = ["InstanSegONNXWrapper", "export_instanseg_onnx"]


class InstanSegONNXWrapper(nn.Module):
    """Return InstanSeg four reconstruction channels and class logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction maps and class logits with BCHW axes."""
        nucleus = self.model(x)["nuc"]
        return nucleus.aux_map, nucleus.type_map


def export_instanseg_onnx(
    model: InstanSeg | nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Save InstanSeg prediction maps as an ONNX file.

    Args:
        model: A trained ``InstanSeg`` instance or its PyTorch network.
        output_path: Destination .onnx file. Parent folders are created;
            an existing file may be overwritten.
        input_shape: Example (batch, channels, height, width) shape. Image size
            stays fixed; a dynamic export traces with at least two images.
        opset_version: ONNX operator-set version, default 18.
        dynamic_batch: Allow the number of images in a batch to vary.

    Returns:
        Path to the exported file. Input ``image`` is floating BCHW. Outputs
        ``instanseg_map`` (four reconstruction channels) and ``type_map`` (class logits)
        use the same axes. Preprocessing and instance reconstruction stay in Python.

    Raises:
        ValueError: Input dimensions are not four positive integers.
        TypeError: The model is not a InstanSeg instance or a PyTorch module.
        ImportError: Export requires the optional onnx and onnxscript packages.
        RuntimeError: The PyTorch version is unsupported or export fails.

    Note:
        Export uses the network's device and floating dtype. Every module's
        training mode is restored after success or failure.
        The attached pixel classifier is not exported; keep its trained weights
        with the Python model used for InstanSeg reconstruction.
    """
    network = model.model if isinstance(model, InstanSeg) else model
    return _export_onnx(
        network,
        InstanSegONNXWrapper,
        ("instanseg_map", "type_map"),
        output_path,
        input_shape,
        opset_version,
        dynamic_batch,
    )
