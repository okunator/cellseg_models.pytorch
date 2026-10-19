"""Shared mechanics for exporting dense segmentation prediction maps."""

from pathlib import Path

import torch
from torch import nn


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
        raise RuntimeError("ONNX export requires PyTorch >= 2.7.")


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


def _export_onnx(
    network: nn.Module,
    wrapper_cls: type[nn.Module],
    output_names: tuple[str, ...],
    output_path: str | Path,
    input_shape: tuple[int, int, int, int],
    opset_version: int,
    dynamic_batch: bool,
) -> Path:
    """Export maps while preserving the network's device, dtype and module modes."""
    if len(input_shape) != 4 or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
        for dim in input_shape
    ):
        raise ValueError("input_shape must contain four positive BCHW dimensions.")

    _check_torch_export_version()
    _check_onnx_export_dependencies()

    if not isinstance(network, nn.Module):
        raise TypeError("model must be a model instance or torch.nn.Module.")

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

    wrapper = wrapper_cls(network)
    training_states = [(module, module.training) for module in network.modules()]
    try:
        wrapper.eval()
        # Keep the network dtype when exporting from an AMP training loop.
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            torch.onnx.export(
                wrapper,
                (example,),
                str(output_path),
                input_names=["image"],
                output_names=list(output_names),
                dynamo=True,
                dynamic_shapes=dynamic_shapes,
                opset_version=opset_version,
            )
    finally:
        for module, training in training_states:
            module.training = training

    return output_path
