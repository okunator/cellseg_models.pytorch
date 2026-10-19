from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.models.base._onnx import _export_onnx

from cellseg_models_pytorch.models.stardist.stardist import StarDist

__all__ = ["StarDistONNXWrapper", "export_stardist_onnx"]


class StarDistONNXWrapper(nn.Module):
    """Flatten StarDist's structured output into ONNX-friendly tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return BCHW foreground scores, ray distances, and class logits."""
        nuc = self.model(x)["nuc"]
        if nuc.binary_map is None:
            raise RuntimeError("StarDist export requires a nuclei binary output head.")

        return nuc.binary_map, nuc.aux_map, nuc.type_map


def export_stardist_onnx(
    model: Union[StarDist, nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export the StarDist neural-network forward pass to ONNX.

    The exported graph contains only the dense neural-network prediction stage.
    StarDist polygon reconstruction and overlapping-object suppression remain in the existing Python
    post-processing pipeline.

    Args:
        model: A high-level ``StarDist`` instance or its underlying ``.model``
            module.
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

        The input is named ``image``; outputs are ``binary_map``, ``ray_map`` and ``type_map``.
        Outputs have batch, channel, height and width axes. Cell-type outputs
        are raw logits. Preprocessing and Python reconstruction are not exported.
    """
    network = model.model if isinstance(model, StarDist) else model
    return _export_onnx(
        network,
        StarDistONNXWrapper,
        ("binary_map", "ray_map", "type_map"),
        output_path,
        input_shape,
        opset_version,
        dynamic_batch,
    )
