from pathlib import Path

import torch
from torch import nn

from cellseg_models_pytorch.models.base._onnx import _export_onnx
from cellseg_models_pytorch.models.hovernet.hovernet import HoverNet

__all__ = ["HoverNetONNXWrapper", "export_hovernet_onnx"]


class HoverNetONNXWrapper(nn.Module):
    """Return HoverNet horizontal/vertical distance maps and class logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction maps and class logits with BCHW axes."""
        nucleus = self.model(x)["nuc"]
        return nucleus.aux_map, nucleus.type_map


def export_hovernet_onnx(
    model: HoverNet | nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Save HoverNet prediction maps as an ONNX file.

    Args:
        model: A trained ``HoverNet`` instance or its PyTorch network.
        output_path: Destination .onnx file. Parent folders are created;
            an existing file may be overwritten.
        input_shape: Example (batch, channels, height, width) shape. Image size
            stays fixed; a dynamic export traces with at least two images.
        opset_version: ONNX operator-set version, default 18.
        dynamic_batch: Allow the number of images in a batch to vary.

    Returns:
        Path to the exported file. Input ``image`` is floating BCHW. Outputs
        ``hv_map`` (horizontal/vertical distance maps) and ``type_map`` (class logits)
        use the same axes. Preprocessing and instance reconstruction stay in Python.

    Raises:
        ValueError: Input dimensions are not four positive integers.
        TypeError: The model is not a HoverNet instance or a PyTorch module.
        ImportError: Export requires the optional onnx and onnxscript packages.
        RuntimeError: The PyTorch version is unsupported or export fails.

    Note:
        Export uses the network's device and floating dtype. Every module's
        training mode is restored after success or failure.
    """
    network = model.model if isinstance(model, HoverNet) else model
    return _export_onnx(
        network,
        HoverNetONNXWrapper,
        ("hv_map", "type_map"),
        output_path,
        input_shape,
        opset_version,
        dynamic_batch,
    )
