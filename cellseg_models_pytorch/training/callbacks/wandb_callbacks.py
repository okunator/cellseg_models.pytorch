from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from skimage.color import label2rgb
from tqdm import tqdm

try:
    import wandb
except ImportError:
    raise ImportError("wandb required. `pip install wandb`")

from ...inference import PostProcessor
from ...metrics.functional import iou_multiclass, panoptic_quality
from ...utils import get_type_instances, remap_label
from ..functional import iou

__all__ = ["WandbImageCallback", "WandbClassLineCallback", "WandbGetExamplesCallback"]


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

    def train_batch_end(
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

    def validation_batch_end(
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

    def test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Log the inputs and outputs of the model to wandb."""
        self.batch_end(trainer, outputs["soft_masks"], batch, batch_idx, phase="test")


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

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Call the callback at val time."""
        self.validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Call the callback at val time."""
        self.train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )


class WandbGetExamplesCallback(pl.Callback):
    def __init__(
        self,
        type_classes: Dict[str, int],
        sem_classes: Optional[Dict[str, int]],
        instance_postproc: str,
        inst_key: str,
        aux_key: str,
        inst_act: str = "softmax",
        aux_act: str = None,
    ) -> None:
        """Create a wandb callback that logs img examples at test time."""
        super().__init__()
        self.type_classes = type_classes
        self.sem_classes = sem_classes
        self.inst_key = inst_key
        self.aux_key = aux_key

        self.inst_act = inst_act
        self.aux_act = aux_act

        self.postprocessor = PostProcessor(
            instance_postproc=instance_postproc, inst_key=inst_key, aux_key=aux_key
        )

    def post_proc(
        self, outputs: Dict[str, torch.Tensor]
    ) -> List[Dict[str, np.ndarray]]:
        """Apply post processing to the outputs."""
        B, _, _, _ = outputs[self.inst_key].shape

        inst = outputs[self.inst_key].detach()
        if self.inst_act == "softmax":
            inst = F.softmax(inst, dim=1)
        if self.inst_act == "sigmoid":
            inst = torch.sigmoid(inst)

        aux = outputs[self.aux_key].detach()
        if self.aux_act == "tanh":
            aux = torch.tanh(aux)

        sem = None
        if "sem" in outputs.keys():
            sem = outputs["sem"].detach()
            sem = F.softmax(sem, dim=1).cpu().numpy()

        typ = None
        if "type" in outputs.keys():
            typ = outputs["type"].detach()
            typ = F.softmax(typ, dim=1).cpu().numpy()

        inst = inst.cpu().numpy()
        aux = aux.cpu().numpy()
        outmaps = []
        for i in range(B):
            maps = {
                self.inst_key: inst[i],
                self.aux_key: aux[i],
            }
            if sem is not None:
                maps["sem"] = sem[i]
            if typ is not None:
                maps["type"] = typ[i]

            out = self.postprocessor.post_proc_pipeline(maps)
            outmaps.append(out)

        return outmaps

    def count_pixels(self, img: np.ndarray, shape: int):
        """Compute pixel proportions per class."""
        return [float(p) / shape**2 for p in np.bincount(img.astype(int).flatten())]

    def epoch_end(self, trainer, pl_module) -> None:
        """Log metrics at the epoch end."""
        decs = [list(it.keys()) for it in pl_module.heads.values()]
        outheads = [item for sublist in decs for item in sublist]

        loader = trainer.datamodule.test_dataloader()
        runid = trainer.logger.experiment.id
        test_res_at = wandb.Artifact("test_pred_" + runid, "test_preds")

        # Create artifact
        runid = trainer.logger.experiment.id
        test_res_at = wandb.Artifact("test_pred_" + runid, "test_preds")

        # Init wb table
        cols = ["id", "inst_gt", "inst_pred", "bPQ"]

        if "type" in outheads:
            cols += [
                "cell_types",
                *[f"pq_{c}" for c in self.type_classes.values() if c != "bg"],
            ]
        if "sem" in outheads:
            cols += [
                "tissue_types",
                *[f"iou_{c}" for c in self.sem_classes.values() if c != "bg"],
            ]

        model_res_table = wandb.Table(columns=cols)

        #
        with tqdm(loader, unit="batch") as loader:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    soft_masks = pl_module.forward(batch["image"].to(pl_module.device))
                    imgs = list(batch["image"].detach().cpu().numpy())  # [(C, H, W)*B]
                    inst_targets = list(batch["inst_map"].detach().cpu().numpy())

                    outmaps = self.post_proc(soft_masks)

                    type_targets = None
                    if "type" in list(batch.keys()):
                        type_targets = list(
                            batch["type"].detach().cpu().numpy()
                        )  # [(C, H, W)*B]

                    sem_targets = None
                    if "sem" in list(batch.keys()):
                        sem_targets = list(
                            batch["sem"].detach().cpu().numpy()
                        )  # [(C, H, W)*B]

                    # loop the images in batch
                    for i, (pred, im, inst_target) in enumerate(
                        zip(outmaps, imgs, inst_targets)
                    ):
                        inst_targ = remap_label(inst_target)
                        inst_pred = remap_label(pred["inst"])

                        wb_inst_gt = wandb.Image(label2rgb(inst_targ, bg_label=0))
                        wb_inst_pred = wandb.Image(label2rgb(inst_pred, bg_label=0))
                        pq_inst = panoptic_quality(inst_targ, inst_pred)["pq"]

                        row = [
                            f"test_batch_{batch_idx}_{i}",
                            wb_inst_gt,
                            wb_inst_pred,
                            pq_inst,
                        ]

                        if type_targets is not None:
                            per_class_pq = [
                                panoptic_quality(
                                    remap_label(
                                        get_type_instances(
                                            inst_targ, type_targets[i], j
                                        )
                                    ),
                                    remap_label(
                                        get_type_instances(inst_pred, pred["type"], j)
                                    ),
                                )["pq"]
                                for j in self.type_classes.keys()
                                if j != 0
                            ]

                            type_classes_set = wandb.Classes(
                                [
                                    {"name": name, "id": id}
                                    for id, name in self.type_classes.items()
                                    if id != 0
                                ]
                            )
                            wb_type = wandb.Image(
                                im.transpose(1, 2, 0),
                                classes=type_classes_set,
                                masks={
                                    "ground_truth": {"mask_data": type_targets[i]},
                                    "pred": {"mask_data": pred["type"]},
                                },
                            )

                            row += [wb_type, *per_class_pq]

                        if sem_targets is not None:
                            per_class_iou = list(
                                iou_multiclass(
                                    sem_targets[i], pred["sem"], len(self.sem_classes)
                                )
                            )

                            sem_classes_set = wandb.Classes(
                                [
                                    {"name": name, "id": id}
                                    for id, name in self.sem_classes.items()
                                    if id != 0
                                ]
                            )
                            wb_sem = wandb.Image(
                                im.transpose(1, 2, 0),
                                classes=sem_classes_set,
                                masks={
                                    "ground_truth": {"mask_data": sem_targets[i]},
                                    "pred": {"mask_data": pred["sem"]},
                                },
                            )
                            row += [wb_sem, *per_class_iou[1:]]

                        model_res_table.add_data(*row)

        test_res_at.add(model_res_table, "model_batch_results")
        trainer.logger.experiment.log_artifact(test_res_at)

    def on_test_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Call the callback at test time."""
        self.epoch_end(trainer, pl_module)
