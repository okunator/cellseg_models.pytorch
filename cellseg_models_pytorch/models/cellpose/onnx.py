from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.models.base._onnx import _export_onnx

from cellseg_models_pytorch.models.cellpose.cellpose import CellPose

__all__ = ["CellPoseONNXWrapper", "export_cellpose_onnx"]


class CellPoseONNXWrapper(nn.Module):
    """Return CellPose or Omnipose flow maps and class scores as two tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flow and type-logit maps."""
        nuc = self.model(x)["nuc"]
        if nuc.type_map is None:
            raise RuntimeError("ONNX export requires a nuclei cell-type output head.")
        return nuc.aux_map, nuc.type_map


def export_cellpose_onnx(
    model: Union[CellPose, nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export CellPose or Omnipose prediction maps to ONNX.

    The exported graph contains only the dense neural-network prediction stage.
    Flow integration and instance reconstruction remain in the existing Python
    post-processing pipeline.

    Args:
        model: A ``CellPose`` instance, its underlying network, or an Omnipose
            network created with ``omnipose_nuclei``. Load trained weights first.
        output_path: Destination ``.onnx`` file. Parent folders are created; an existing
                file may be overwritten.
        input_shape: Example (batch, channels, height, width) input shape.
            Spatial dimensions are fixed in the exported graph. With dynamic batches,
            the tracing batch is at least two to avoid specializing batch one.
            Defaults to ``(1, 3, 256, 256)``.
        opset_version: ONNX opset version. Defaults to 18.
        dynamic_batch: Mark the input batch dimension dynamic. Output batch
            dimensions inherit the same symbolic dimension through the exported
            graph. Defaults to True.

    Returns:
        Path to the exported ONNX model.

    Raises:
        ValueError: input_shape does not contain four positive integer dimensions.
        TypeError: model is not the expected wrapper or a PyTorch module.
        ImportError: onnx or onnxscript is unavailable.
        RuntimeError: The PyTorch version is too old or export fails.

    Note:
        Requires onnx and onnxscript, in addition to the package dependencies.
        The exporter checks for PyTorch 2.7 or later; the package's own minimum
        PyTorch version still applies. Training/evaluation state of every network
        module is restored on success or failure. Example inputs use the network's
        device and floating dtype; exporting does not move the model.

        The input is named ``image``; outputs are ``flow_map`` and ``type_map``.
        Outputs have batch, channel, height and width axes. Cell-type outputs
        are raw logits. Preprocessing and Python reconstruction are not exported.
    """
    network = model.model if isinstance(model, CellPose) else model
    return _export_onnx(
        network,
        CellPoseONNXWrapper,
        ("flow_map", "type_map"),
        output_path,
        input_shape,
        opset_version,
        dynamic_batch,
    )
