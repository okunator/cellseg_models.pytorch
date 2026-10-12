"""Train binary nuclei classification and CellPose flows on paired RGB/.mat patches.

See the Training section of README.md for data layout and installation.
"""

import argparse
from pathlib import Path

import albumentations as A
import torch
from torch.utils.data import DataLoader

from cellseg_models_pytorch.losses import CELoss, MSE, JointLoss, MultiTaskLoss
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.torch_datasets import TrainDatasetFolder
from cellseg_models_pytorch.transforms.albu_transforms import (
    ApplyEach,
    CellposeTransform,
)


def train(
    images: Path,
    labels: Path,
    checkpoint: Path,
    epochs: int = 1,
    device: str = "cpu",
) -> None:
    """Train from scratch and save model, optimizer and RNG state.

    Args:
        images: Folder containing corresponding RGB patches of the same size.
        labels: Folder of .mat files containing integer (H, W) `inst` masks.
            Zero is background; positive IDs identify individual nuclei.
        checkpoint: New output file; existing files are never overwritten.
        epochs: Positive number of passes through the dataset.
        device: Explicit PyTorch device, such as "cpu" or "cuda".

    Raises:
        ValueError: If epochs or patch dimensions are invalid, or data is empty.
        FileExistsError: If the checkpoint already exists.
        RuntimeError: If the training loss is nonfinite.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.manual_seed(0)
    target_device = torch.device(device)
    dataset = TrainDatasetFolder(
        path_img=str(images),
        path_label=str(labels),
        mask_keys=("inst",),
        transforms=A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))]),
        inst_transforms=ApplyEach([CellposeTransform()], as_list=True),
    )
    if len(dataset) == 0:
        raise ValueError("training folders must contain paired patches")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    model_kwargs = {
        "n_nuc_classes": 2,
        "enc_name": "resnet18",
        "enc_pretrain": False,
        "out_channels": (64, 32, 16, 8),
        "layer_depths": (1, 1, 1, 1),
    }
    model = cellpose_nuclei(**model_kwargs).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss(
        head_losses={
            "nuc_type": JointLoss([CELoss()]),
            "nuc_cellpose": JointLoss([MSE()]),
        }
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            image = batch["image"].to(target_device)
            if any(size < 64 or size % 32 for size in image.shape[-2:]):
                raise ValueError("patch height/width must be multiples of 32 and >=64")
            targets = {
                "nuc_type": (batch["inst"] > 0).long().to(target_device),
                "nuc_cellpose": batch["cellpose_inst"].float().to(target_device),
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(image)["nuc"]
            loss = criterion(
                {"nuc_type": output.type_map, "nuc_cellpose": output.aux_map}, targets
            )
            if not torch.isfinite(loss):
                raise RuntimeError("training loss is nonfinite")
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * image.shape[0]
        print(f"Epoch {epoch + 1}: loss={total_loss / len(dataset):.6f}")
    with checkpoint.open("xb") as file:
        torch.save(
            {
                "model": model.state_dict(),
                "model_kwargs": model_kwargs,
                "optimizer": optimizer.state_dict(),
                "epoch": epochs,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if target_device.type == "cuda"
                else [],
            },
            file,
        )


def main() -> None:
    """Parse paths and train without downloading weights or configuring a framework."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.images, args.labels, args.checkpoint, args.epochs, args.device)


if __name__ == "__main__":
    main()
