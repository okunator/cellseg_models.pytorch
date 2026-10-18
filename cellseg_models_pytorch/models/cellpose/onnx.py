from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

from cellseg_models_pytorch.models.cellpose.cellpose import CellPose

__all__ = ["CellPoseONNXWrapper", "export_cellpose_onnx"]


class CellPoseONNXWrapper(nn.Module):
    """Flatten CellPose's structured output into ONNX-friendly tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return flow and type-logit maps."""
        nuc = self.model(x)["nuc"]
        return nuc.aux_map, nuc.type_map


def _check_torch_export_version() -> None:
    """Require the validated torch.export-based ONNX exporter (PyTorch 2.7)."""
    try:
        major, minor = (
            int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Unable to determine PyTorch version from {torch.__version__!r}."
        ) from None

    if (major, minor) < (2, 7):
        raise RuntimeError("CellPose ONNX export requires PyTorch >= 2.7.")


def _check_onnx_export_dependencies() -> None:
    """Raise an actionable error when PyTorch ONNX dependencies are missing."""
    missing = []
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    try:
        import onnxscript  # noqa: F401
    except ImportError:
        missing.append("onnxscript")

    if missing:
        packages = " ".join(missing)
        raise ImportError(
            "ONNX export requires additional packages. "
            f"Install them with `pip install {packages}`."
        )


def export_cellpose_onnx(
    model: Union[CellPose, nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export the CellPose neural-network forward pass to ONNX.

    The exported graph contains only the dense neural-network prediction stage.
    Flow integration and instance reconstruction remain in the existing Python
    post-processing pipeline.

    Args:
        model: A high-level ``CellPose`` instance or its underlying ``.model``
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

        The input is named ``image``; outputs are ``flow_map`` and ``type_map``.
        Outputs have batch, channel, height and width axes. Cell-type outputs
        are raw logits. Preprocessing and Python reconstruction are not exported.
    """
    if len(input_shape) != 4 or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
        for dim in input_shape
    ):
        raise ValueError("input_shape must contain four positive BCHW dimensions.")

    _check_torch_export_version()
    _check_onnx_export_dependencies()

    network = model.model if isinstance(model, CellPose) else model
    if not isinstance(network, nn.Module):
        raise TypeError("model must be a CellPose instance or torch.nn.Module.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference = next(network.parameters(), None)
    if reference is None:
        reference = next(network.buffers(), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = (
        reference.dtype
        if reference is not None and reference.is_floating_point()
        else torch.get_default_dtype()
    )

    # Batch-one examples are specialized by torch.export even with a dynamic Dim.
    example_shape = (
        (max(2, input_shape[0]), *input_shape[1:]) if dynamic_batch else input_shape
    )
    example = torch.zeros(example_shape, dtype=dtype, device=device)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}

    wrapper = CellPoseONNXWrapper(network)
    training_states = [(module, module.training) for module in network.modules()]
    try:
        wrapper.eval()
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (example,),
                str(output_path),
                input_names=["image"],
                output_names=["flow_map", "type_map"],
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                opset_version=opset_version,
            )
    finally:
        for module, training in training_states:
            module.training = training

    return output_path
