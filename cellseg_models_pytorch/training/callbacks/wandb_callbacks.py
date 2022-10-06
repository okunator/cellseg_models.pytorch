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

__all__ = ["WandbImageCallback", "WandbClassMetricCallback"]


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


class WandbClassMetricCallback(pl.Callback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        freq: int = 100,
        return_series: bool = True,
        return_bar: bool = True,
        return_table: bool = False,
    ) -> None:
        """Call back to compute per-class ious and log them to wandb."""
        super().__init__()
        self.type_classes = type_classes
        self.sem_classes = sem_classes
        self.freq = freq
        self.return_series = return_series
        self.return_bar = return_bar
        self.return_table = return_table
        self.cell_ious = np.empty(0)
        self.sem_ious = np.empty(0)

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

    def get_table(
        self, ious: np.ndarray, x: np.ndarray, classes: Dict[int, str]
    ) -> wandb.Table:
        """Return a wandb Table with step, iou and label values for every step."""
        batch_data = [
            [xi * self.freq, c, np.round(ious[xi, i], 4)]
            for i, c, in classes.items()
            for xi in x
        ]

        return wandb.Table(data=batch_data, columns=["step", "label", "value"])

    def get_bar(self, iou: np.ndarray, classes: Dict[int, str], title: str) -> Any:
        """Return a wandb bar plot object of the current per class iou values."""
        batch_data = [[lab, val] for lab, val in zip(list(classes.values()), iou)]
        table = wandb.Table(data=batch_data, columns=["label", "value"])
        return wandb.plot.bar(table, "label", "value", title=title)

    def get_series(
        self, ious: np.ndarray, x: np.ndarray, classes: Dict[int, str], title: str
    ) -> Any:
        """Return a wandb series plot obj of the per class iou values over timesteps."""
        return wandb.plot.line_series(
            xs=x.tolist(),
            ys=[ious[:, c].tolist() for c in classes.keys()],
            keys=list(classes.values()),
            title=title,
            xname="step",
        )

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
                self.cell_ious = np.append(self.cell_ious, iou)
                cell_ious = self.cell_ious.reshape(-1, len(self.type_classes))
                x = np.arange(cell_ious.shape[0])

                if self.return_table:
                    log_dict[f"{phase}/type_ious_table"] = self.get_table(
                        cell_ious, x, self.type_classes
                    )

                if self.return_series:
                    log_dict[f"{phase}/type_ious_per_class"] = self.get_series(
                        cell_ious, x, self.type_classes, title="Per type class mIoU"
                    )

                if self.return_bar:
                    log_dict[f"{phase}/type_ious_bar"] = self.get_bar(
                        list(iou), self.type_classes, title="Cell class mIoUs"
                    )

            if "sem" in list(batch.keys()):
                iou = self.compute("sem", outputs, batch)

                self.sem_ious = np.append(self.sem_ious, iou)
                sem_ious = self.sem_ious.reshape(-1, len(self.sem_classes))
                x = np.arange(sem_ious.shape[0])

                if self.return_table:
                    log_dict[f"{phase}/sem_ious_table"] = self.get_table(
                        cell_ious, x, self.type_classes
                    )

                if self.return_series:
                    log_dict[f"{phase}/sem_ious_per_class"] = self.get_series(
                        cell_ious, x, self.type_classes, title="Per sem class mIoU"
                    )

                if self.return_bar:
                    log_dict[f"{phase}/sem_ious_bar"] = self.get_bar(
                        list(iou), self.type_classes, title="Sem class mIoUs"
                    )

            trainer.logger.experiment.log(log_dict)

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
