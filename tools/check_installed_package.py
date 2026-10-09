"""Exercise the installed distribution outside the source checkout."""

import importlib
import sys
from importlib.metadata import version
from pathlib import Path

import torch

import cellseg_models_pytorch
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei


def main() -> None:
    """Check package identity, public imports, and a CPU forward/backward pass."""
    installed_path = Path(cellseg_models_pytorch.__file__).resolve()
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected an installed package, got {installed_path}")
    if version("cellseg_models_pytorch") != cellseg_models_pytorch.__version__:
        raise RuntimeError("Installed metadata and module versions disagree")

    for module in (
        "models.cellpose",
        "models.stardist",
        "models.hovernet",
        "models.cellvit",
        "models.cppnet",
        "models.instanseg",
        "metrics",
        "losses",
        "inference",
        "utils",
        "wsi",
    ):
        importlib.import_module(f"cellseg_models_pytorch.{module}")

    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = cellpose_nuclei(3, enc_name="resnet18", enc_pretrain=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    output = model(torch.rand(2, 3, 64, 64))["nuc"]
    if output.type_map.shape != (2, 3, 64, 64):
        raise RuntimeError(f"Unexpected type output: {output.type_map.shape}")
    if output.aux_map.shape != (2, 2, 64, 64):
        raise RuntimeError(f"Unexpected flow output: {output.aux_map.shape}")
    loss = output.type_map.square().mean() + output.aux_map.square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("CPU training loss is not finite")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("CPU model gradients are missing or nonfinite")
    optimizer.step()
    print(f"Validated {installed_path} with a CPU training step")


if __name__ == "__main__":
    main()
