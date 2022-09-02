from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the `SegmentationExperiment`, pytorch-lightning is required. "
        "Install with `pip install pytorch-lightning`"
    )

from ...losses import JOINT_SEG_LOSSES, SEG_LOSS_LOOKUP, JointLoss, Loss, MultiTaskLoss
from ...optimizers import OPTIM_LOOKUP, SCHED_LOOKUP, adjust_optim_params
from ..callbacks import METRIC_LOOKUP


class SegmentationExperiment(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        branch_losses: Dict[str, int],
        branch_metrics: Dict[str, List[str]] = None,
        edge_weights: Dict[str, float] = None,
        class_weights: Dict[str, bool] = None,
        optimizer: str = "adam",
        lookahead: bool = False,
        dec_learning_rate: float = 0.0005,
        enc_learning_rate: float = 0.00005,
        dec_weight_decay: float = 0.0003,
        enc_weight_decay: float = 0.00003,
        rm_bias_weight_decay: bool = True,
        scheduler: str = "reduce_on_plateau",
        scheduler_params: Dict[str, Any] = None,
    ) -> None:
        """Segmentation model training experiment.

        Configures the training setup for a segmentation model.

        Parameters
        ----------
            model : nn.Module
                A segmentation model specification.
            branch_losses : Dict[str, str]
                A Dict of branch names mapped to a string specifying a jointloss.
                E.g. {"inst": "tversky_ce_ssim", "sem": "mse"}. Allowed losses: "mse",
                "ce", "sce", "ssim", "msssim", "tversky", "focal", "iou", "dice".
                Use underscores to create joint loss functions. e.g. "dice_ce_tversky".
            branch_metrics : Dict[str, List[str]], optional
                A Dict of branch names mapped to a list of strings specifying a metrics.
                E.g. {"inst": ["acc"], "sem": ["miou", "acc"]}. Allowed metrics: "miou",
                "acc". If None, no metrics will be recorded during training.
            edge_weights : Dict[str, float], optional
                A dictionary of branch names mapped to floats that are used to weight
                nuclei edges in CE-based losses. E.g. {"inst": 1.1, "sem": None}.
                If None, no edge weights will be used during training.
            class_weights : Dict[str, torch.Tensor], optional
                A dictionary of branch names mapped to class weight tensors
                of shape (n_classes_branch, ). E.g. {"inst": tensor([[0.4, 0.6]])}.
                If None, no edge weights will be used during training.
            optimizer : str, default="adam"
                Name of the optimizer. In-built optimizers from `torch` and
                `torch_optimizer` packages can be used. One of: "adam", "rmsprop","sgd",
                "adadelta", "apollo", "adabelief", "adamp", "adagrad", "adamax",
                "adamw", "asdg", "accsgd", "adabound", "adamod", "diffgrad", "lamb",
                "novograd", "pid", "qhadam", "qhm", "radam", "sgdw", "yogi", "ranger",
                "rangerqh","rangerva"
            lookahead : bool, default=False
                Flag whether the optimizer uses lookahead.
            dec_learning_rate : float, default=0.0005
                Learning rate for the decoder(s) during training.
            enc_learning_rate : float, default=0.00005
                Learning rate for the encoder during training.
            dec_weight_decay : float, defauilt=0.0003
                Weight decay for the decoder(s) during training.
            enc_weight_decay : float, default=0.00003
                Weight decay for the encoder during training.
            rm_bias_weight_decay : bool, default=True
                Flag whether to remove weight decay for bias terms.
            scheduler : str, default="reduce_on_plateau"
                The name of the learning rate scheduler (torch in-built schedulers).
                Allowed ones: "reduce_on_plateau", "lambda", "cyclic", "exponential",
                "cosine_annealing", "cosine_annealing_warm".
            scheduler_params : Dict[str, Any]
                Params dict for the scheduler. Refer to torch lr_scheduler docs
                for the possible scheduler arguments.

        Raises
        ------
            ValueError if decoder branch related dicts don't have matching keys.
            ValueError if `branch_losses` contains illegal joint loss names.
            ValueError if illegal metric names are given.
            ValueError if illegal optimizer name is given.
            ValueError if illegal scheduler name is given.
        """
        super().__init__()
        self.model = model
        self.heads = model.heads
        self.aux_key = model.aux_key
        self.inst_key = model.inst_key

        # Optimizer args
        self.optimizer = optimizer
        self.dec_learning_rate = dec_learning_rate
        self.enc_learning_rate = enc_learning_rate
        self.dec_weight_decay = dec_weight_decay
        self.enc_weight_decay = enc_weight_decay
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.lookahead = lookahead
        self.rm_bias_weight_decay = rm_bias_weight_decay

        # loss/metircs args
        self.branch_losses = branch_losses
        self.branch_metrics = branch_metrics
        self.edge_weights = edge_weights
        self.class_weights = class_weights

        self._validate_branch_args()
        self.save_hyperparameters(ignore="model")

        self.criterion = self.configure_loss()
        metrics = self.configure_metrics()
        self.train_metrics = deepcopy(metrics)
        self.val_metrics = deepcopy(metrics)
        self.test_metrics = deepcopy(metrics)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward."""
        return self.model(x)

    def step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, phase: str
    ) -> Dict[str, torch.Tensor]:
        """General step."""
        soft_masks = self.model(batch["image"])
        targets = {k: val for k, val in batch.items() if k != "image"}

        loss = self.criterion(yhats=soft_masks, targets=targets)
        metrics = self.compute_metrics(soft_masks, targets, phase)

        ret = {
            "soft_masks": soft_masks if phase == "val" else None,
            "loss": loss,
        }

        return {**ret, **metrics}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step + train metric logs."""
        res = self.step(batch, batch_idx, "train")

        del res["soft_masks"]  # soft masks not needed for logging
        loss = res.pop("loss")

        # log all the metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=False, on_step=True)
        for k, val in res.items():
            self.log(f"train_{k}", val, prog_bar=True, on_epoch=False, on_step=True)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validate step + validation metric logs + example outputs for logging."""
        res = self.step(batch, batch_idx, "val")

        soft_masks = res.pop("soft_masks")  # soft masks for logging
        loss = res.pop("loss")

        # log all the metrics
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        for k, val in res.items():
            self.log(f"val_{k}", val, prog_bar=False, on_epoch=True, on_step=False)

        # If batch_idx in (0, 10, 50), sends outputs to logger
        if batch_idx in (0, 10, 50):
            return soft_masks

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step + test metric logs."""
        res = self.step(batch, batch_idx, "test")

        del res["soft_masks"]  # soft masks not needed for logging
        loss = res.pop("loss")

        # log all the metrics
        self.log("test_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        for k, val in res.items():
            self.log(f"test_{k}", val, prog_bar=False, on_epoch=True, on_step=False)

    def compute_metrics(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: str,
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics for logging."""
        if phase == "train":
            metrics_dict = self.train_metrics
        elif phase == "val":
            metrics_dict = self.val_metrics
        elif phase == "test":
            metrics_dict = self.test_metrics

        ret = {}
        for metric_name, metric in metrics_dict.items():
            if metric is not None:
                branch = metric_name.split("_")[0]
                act = "softmax" if branch in ("inst", "type", "sem") else None
                # act = None if branch in ("inst", "type", "sem") else "softmax"
                ret[branch] = metric(preds[branch], targets[branch], act)

        return ret

    def _validate_branch_args(self) -> None:
        """Check that there are no conflicting decoder branch args."""
        lk = set(self.branch_losses.keys())
        dk = set(self.model._get_inner_keys(self.model.heads))
        has_same_keys = lk == dk

        mk = None
        if self.branch_metrics is not None:
            mk = set(self.branch_metrics.keys())
            has_same_keys = dk == lk == mk

        ek = None
        if self.edge_weights is not None:
            ek = set(self.edge_weights.keys())
            has_same_keys = dk == lk == mk == ek

        ck = None
        if self.class_weights is not None:
            ck = set(self.class_weights.keys())
            has_same_keys = dk == lk == mk == ek == ck

        if not has_same_keys:
            raise ValueError(
                "Got mismatching keys for branch dict args. "
                f"Branch losses: {lk}. "
                f"Decoder branches: {dk}. "
                f"Metrics: {mk}. "
                f"Edge weights: {ek}. "
                f"Class weights: {ck}. "
                f"(Metrics, edge weights, and class weights can be None)"
            )

    def configure_loss(self) -> nn.Module:
        """Configure the single/multi-task loss for the model."""
        branch_losses = {}
        for branch, loss in self.branch_losses.items():
            if loss not in JOINT_SEG_LOSSES:
                raise ValueError(
                    f"Illegal joint loss given. Got: {loss}. Allowed are all the "
                    f"permutations of: {list(SEG_LOSS_LOOKUP.keys())}"
                )

            try:
                losses = loss.split("_")
            except Exception:
                losses = [loss]

            # set edge and class weights
            edge_weight = None
            if self.edge_weights is not None:
                edge_weight = self.edge_weights[branch]

            class_weight = None
            if self.class_weights:
                class_weight = self.class_weights[branch]

            # set joint losses for each branch
            branch_losses[branch] = JointLoss(
                [
                    Loss(loss_name, edge_weight=edge_weight, class_weights=class_weight)
                    for loss_name in losses
                ]
            )

        return MultiTaskLoss(branch_losses=branch_losses)

    def configure_metrics(self):
        """Configure the train, val, test metrics for logging."""
        allowed = list(METRIC_LOOKUP.keys()) + [None]
        metrics = nn.ModuleDict()
        for k, m in self.branch_metrics.items():
            for metric_name in m:
                if metric_name not in allowed:
                    raise ValueError(
                        f"Illegal metric given. Got: {metric_name}. Allowed: {allowed}."
                    )

                if metric_name is not None:
                    metric = METRIC_LOOKUP[metric_name]()
                else:
                    metric = None

                metrics[f"{k}_{metric_name}"] = metric

        return metrics

    def configure_optimizers(self):
        """Configure the optimizer and scheduler."""
        allowed = list(OPTIM_LOOKUP.keys())
        if self.optimizer not in allowed:
            raise ValueError(
                f"Illegal optimizer given. Got {self.optimizer}. Allowed: {allowed}."
            )

        allowed = list(SCHED_LOOKUP.keys())
        if self.scheduler not in allowed:
            raise ValueError(
                f"Illegal scheduler given. Got {self.scheduler}. Allowed: {allowed}."
            )

        params = adjust_optim_params(
            self.model,
            encoder_lr=self.enc_learning_rate,
            encoder_wd=self.enc_weight_decay,
            decoder_lr=self.dec_learning_rate,
            decoder_wd=self.dec_weight_decay,
            remove_bias_wd=self.rm_bias_weight_decay,
        )
        optimizer = OPTIM_LOOKUP[self.optimizer](params)

        if self.lookahead:
            optimizer = OPTIM_LOOKUP["lookahead"](optimizer, k=5, alpha=0.5)

        if self.scheduler_params is None:
            self.scheduler_params = {}

        scheduler = {
            "scheduler": SCHED_LOOKUP[self.scheduler](
                optimizer, **self.scheduler_params
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
