from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    raise ImportError("wandb required. `pip install wandb`")

from ..functional import iou

__all__ = ["WandbImageCallback", "WandbClassBarCallback", "WandbClassLineCallback"]


class WandbImageCallback(pl.Callback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        freq: int = 100,
    ) -> None:
        """Create a callback that logs prediction masks to wandb."""
        super().__init__()
        self.freq = freq
        self.type_classes = type_classes
        self.sem_classes = sem_classes

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Log the inputs and outputs of the model to wandb."""
        if batch_idx % self.freq == 0:
            outputs = outputs["soft_masks"]

            log_dict = {
                "global_step": trainer.global_step,
                "epoch": trainer.current_epoch,
            }

            img = batch["image"].detach().to("cpu").numpy()

            if "type" in list(batch.keys()):
                type_target = batch["type"].detach().to("cpu").numpy()
                soft_types = outputs["type"].detach().to("cpu")
                types = torch.argmax(F.softmax(soft_types, dim=1), dim=1).numpy()

                log_dict["val/cell_types"] = [
                    wandb.Image(
                        im.transpose(1, 2, 0),
                        masks={
                            "predictions": {
                                "mask_data": t,
                                "class_labels": self.type_classes,
                            },
                            "ground_truth": {
                                "mask_data": tt,
                                "class_labels": self.type_classes,
                            },
                        },
                    )
                    for im, t, tt in zip(img, types, type_target)
                ]

            if "sem" in list(batch.keys()):
                sem_target = batch["sem"].detach().to("cpu").numpy()
                soft_sem = outputs["sem"].detach().to(device="cpu")
                sem = torch.argmax(F.softmax(soft_sem, dim=1), dim=1).numpy()

                log_dict["val/tissue_areas"] = [
                    wandb.Image(
                        im.transpose(1, 2, 0),
                        masks={
                            "predictions": {
                                "mask_data": s,
                                "class_labels": self.sem_classes,
                            },
                            "ground_truth": {
                                "mask_data": st,
                                "class_labels": self.sem_classes,
                            },
                        },
                    )
                    for im, s, st in zip(img, sem, sem_target)
                ]

            for m in list(batch.keys()):
                if m not in ("sem", "type", "inst", "image"):
                    aux = outputs[m].detach().to(device="cpu")
                    log_dict[f"val/{m}"] = [
                        wandb.Image(a[i, ...], caption=f"{m} maps")
                        for a in aux
                        for i in range(a.shape[0])
                    ]

            trainer.logger.experiment.log(log_dict)


class WandbIoUCallback(pl.Callback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        freq: int = 100,
    ) -> None:
        """Create a base class for IoU wandb callbacks."""
        super().__init__()
        self.type_classes = type_classes
        self.sem_classes = sem_classes
        self.freq = freq

    def batch_end(self) -> None:
        """Abstract batch end method."""
        raise NotImplementedError

    def compute(
        self,
        key: str,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Compute the iou per class."""
        target = batch[key].detach()
        soft_types = outputs[key].detach()
        pred = F.softmax(soft_types, dim=1)

        met = iou(pred, target).mean(dim=0)
        return met.to("cpu").numpy()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Log the inputs and outputs of the model to wandb."""
        self.batch_end(trainer, outputs["soft_masks"], batch, batch_idx, phase="train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Log the inputs and outputs of the model to wandb."""
        self.batch_end(trainer, outputs["soft_masks"], batch, batch_idx, phase="val")


class WandbClassBarCallback(WandbIoUCallback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        freq: int = 100,
    ) -> None:
        """Create a wandb callback that logs per-class mIoU at batch ends."""
        super().__init__(type_classes, sem_classes, freq)

    def get_bar(self, iou: np.ndarray, classes: Dict[int, str], title: str) -> Any:
        """Return a wandb bar plot object of the current per class iou values."""
        batch_data = [[lab, val] for lab, val in zip(list(classes.values()), iou)]
        table = wandb.Table(data=batch_data, columns=["label", "value"])
        return wandb.plot.bar(table, "label", "value", title=title)

    def batch_end(
        self,
        trainer: pl.Trainer,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        phase: str,
    ) -> None:
        """Log metrics at every 100th step to wandb."""
        if batch_idx % self.freq == 0:
            log_dict = {}
            if "type" in list(batch.keys()):
                iou = self.compute("type", outputs, batch)
                log_dict[f"{phase}/type_ious_bar"] = self.get_bar(
                    list(iou), self.type_classes, title="Cell class mIoUs"
                )

            if "sem" in list(batch.keys()):
                iou = self.compute("sem", outputs, batch)
                log_dict[f"{phase}/sem_ious_bar"] = self.get_bar(
                    list(iou), self.sem_classes, title="Sem class mIoUs"
                )

            trainer.logger.experiment.log(log_dict)


class WandbClassLineCallback(WandbIoUCallback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        freq: int = 100,
    ) -> None:
        """Create a wandb callback that logs per-class mIoU at batch ends."""
        super().__init__(type_classes, sem_classes, freq)

    def get_points(self, iou: np.ndarray, classes: Dict[int, str]) -> Any:
        """Return a wandb bar plot object of the current per class iou values."""
        return {lab: val for lab, val in zip(list(classes.values()), iou)}

    def batch_end(
        self,
        trainer: pl.Trainer,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        phase: str,
    ) -> None:
        """Log metrics at every 100th step to wandb."""
        if batch_idx % self.freq == 0:
            log_dict = {}
            if "type" in list(batch.keys()):
                iou = self.compute("type", outputs, batch)
                log_dict[f"{phase}/type_ious_points"] = self.get_points(
                    list(iou), self.type_classes
                )

            if "sem" in list(batch.keys()):
                iou = self.compute("sem", outputs, batch)
                log_dict[f"{phase}/sem_ious_points"] = self.get_points(
                    list(iou), self.sem_classes
                )

            trainer.logger.experiment.log(log_dict)
