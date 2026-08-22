from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn

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
        """Return binary probability, ray-distance, and type maps."""
        nuc = self.model(x)["nuc"]
        if nuc.binary_map is None:
            raise RuntimeError("StarDist export requires a nuclei binary output head.")

        return nuc.binary_map, nuc.aux_map, nuc.type_map


def _check_torch_export_version() -> None:
    """Require the torch.export-based ONNX exporter introduced in PyTorch 2.5."""
    try:
        major, minor = (
            int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Unable to determine PyTorch version from {torch.__version__!r}."
        ) from None

    if (major, minor) < (2, 5):
        raise RuntimeError("StarDist ONNX export requires PyTorch >= 2.5.")


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


def export_stardist_onnx(
    model: Union[StarDist, nn.Module],
    output_path: Union[str, Path],
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export the StarDist neural-network forward pass to ONNX.

    The exported graph contains only the dense neural-network prediction stage.
    StarDist polygon reconstruction and NMS remain in the existing Python
    post-processing pipeline.

    Parameters
    ----------
    model : StarDist or torch.nn.Module
        A high-level ``StarDist`` instance or its underlying ``.model`` module.
    output_path : str or pathlib.Path
        Destination ``.onnx`` file.
    input_shape : tuple of int, default=(1, 3, 256, 256)
        Example BCHW tensor shape used while exporting the model. Spatial dimensions
        are fixed in the exported graph.
    opset_version : int, default=18
        ONNX opset version.
    dynamic_batch : bool, default=True
        Mark the input batch dimension dynamic. Output batch dimensions inherit the
        same symbolic dimension through the exported graph.

    Returns
    -------
    pathlib.Path
        Path to the exported ONNX model.
    """
    if len(input_shape) != 4 or any(dim <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain four positive BCHW dimensions.")

    _check_torch_export_version()
    _check_onnx_export_dependencies()

    network = model.model if isinstance(model, StarDist) else model
    if not isinstance(network, nn.Module):
        raise TypeError("model must be a StarDist instance or torch.nn.Module.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        device = next(network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    wrapper = StarDistONNXWrapper(network)
    was_training = network.training
    wrapper.eval()

    example = torch.zeros(input_shape, dtype=torch.float32, device=device)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}

    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (example,),
                str(output_path),
                input_names=["image"],
                output_names=["binary_map", "ray_map", "type_map"],
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                opset_version=opset_version,
            )
    finally:
        network.train(was_training)

    return output_path
