from copy import deepcopy
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the `SegmentationExperiment`, lightning is required. "
        "Install with `pip install lightning`"
    )
try:
    from torchmetrics import (
        Dice,
        JaccardIndex,
        MeanSquaredError,
        StructuralSimilarityIndexMeasure,
        UniversalImageQualityIndex,
    )

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`torchmetrics` package is required when using metric callbacks. "
        "Install with `pip install torchmetrics`"
    )


from ...losses import JOINT_SEG_LOSSES, SEG_LOSS_LOOKUP, JointLoss, Loss, MultiTaskLoss
from ...optimizers import OPTIM_LOOKUP, SCHED_LOOKUP, adjust_optim_params

METRIC_LOOKUP = {
    "jaccard": JaccardIndex,
    "dice": Dice,
    "mse": MeanSquaredError,
    "ssim": StructuralSimilarityIndexMeasure,
    "iqi": UniversalImageQualityIndex,
}


class SegmentationExperiment(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        branch_losses: Dict[str, int],
        branch_loss_params: Dict[str, Dict[str, Any]] = None,
        branch_metrics: Dict[str, List[str]] = None,
        optimizer: str = "adam",
        lookahead: bool = False,
        optim_params: Dict[str, Dict[str, Any]] = None,
        scheduler: str = "reduce_on_plateau",
        scheduler_params: Dict[str, Any] = None,
        log_freq: int = 100,
        **kwargs,
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
            branch_loss_params : Dict[str, Dict[str, Any]], optional
                Params for the different losses at different branches. For example to
                use label smoothing or class weighting when computing the losses.
                E.g. {"inst": {"apply_ls": True}, "sem": {"edge_weight": False}}
            branch_metrics : Dict[str, List[str]], optional
                A Dict of branch names mapped to a list of strings specifying a metrics.
                E.g. {"inst": ["acc"], "sem": ["miou", "acc"]}. Allowed metrics: "miou",
                "acc", "ssim", "iqi", "mse", None.
            optimizer : str, default="adam"
                Name of the optimizer. In-built optimizers from `torch` and
                `torch_optimizer` packages can be used. One of: "adam", "rmsprop","sgd",
                "adadelta", "apollo", "adabelief", "adamp", "adagrad", "adamax",
                "adamw", "asdg", "accsgd", "adabound", "adamod", "diffgrad", "lamb",
                "novograd", "pid", "qhadam", "qhm", "radam", "sgdw", "yogi", "ranger",
                "rangerqh","rangerva"
            optim_params : Dict[str, Dict[str, Any]]
                optim paramas like learning rates, weight decays etc for diff parts of
                the network.
                E.g. {"encoder": {"weight_decay": 0.1, "lr": 0.1}, "sem": {"lr": 0.01}}
                or {"learning_rate": 0.005, "weight_decay": 0.03}
            lookahead : bool, default=False
                Flag whether the optimizer uses lookahead.
            scheduler : str, default="reduce_on_plateau"
                The name of the learning rate scheduler (torch in-built schedulers).
                Allowed ones: "reduce_on_plateau", "lambda", "cyclic", "exponential",
                "cosine_annealing", "cosine_annealing_warm".
            scheduler_params : Dict[str, Any]
                Params dict for the scheduler. Refer to torch lr_scheduler docs
                for the possible scheduler arguments.
            log_freq : int, default=100
                Return logs every n batches in logging callbacks.

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

        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.lookahead = lookahead

        self.branch_losses = branch_losses
        self.branch_metrics = branch_metrics
        self.branch_loss_params = branch_loss_params
        self.log_freq = log_freq

        self._validate_branch_args()
        self.save_hyperparameters(ignore="model")

        self.criterion = self.configure_loss()
        metrics = self.configure_metrics()
        self.train_metrics = deepcopy(metrics)
        self.val_metrics = deepcopy(metrics)
        self.test_metrics = deepcopy(metrics)

    @classmethod
    def from_yaml(cls, model: nn.Module, yaml_path: str) -> pl.LightningModule:
        """Initialize the experiment from a yaml-file.

        Parameters
        ----------
            model : nn.Module
                Initialized model.
            yaml_path : str
                Path to the yaml file containing rest of the params
        """
        with open(yaml_path, "r") as stream:
            kwargs = yaml.full_load(stream)

        return cls(model, **kwargs)

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

        if batch_idx % self.log_freq == 0:
            ret_masks = soft_masks
        elif phase == "test":
            ret_masks = soft_masks
        else:
            ret_masks = None

        ret = {
            "soft_masks": ret_masks,
            "loss": loss,
        }

        return {**ret, **metrics}

    def log_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, phase: str
    ) -> Dict[str, torch.Tensor]:
        """Do the logging."""
        on_epoch = phase in ("val", "test")
        on_step = phase == "train"
        prog_bar = phase == "train"

        res = self.step(batch, batch_idx, phase)

        # log all the metrics
        self.log(
            f"{phase}_loss",
            res["loss"],
            prog_bar=prog_bar,
            on_epoch=on_epoch,
            on_step=on_step,
        )

        for k, val in res.items():
            if k not in ("loss", "soft_masks"):
                self.log(
                    f"{phase}_{k}",
                    val,
                    prog_bar=prog_bar,
                    on_epoch=on_epoch,
                    on_step=on_step,
                )

        return res

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step + train metric logs."""
        return self.log_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validate step + validation metric logs + example outputs for logging."""
        return self.log_step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step + test metric logs."""
        return self.log_step(batch, batch_idx, "test")

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
                ret[metric_name] = metric(preds[branch], targets[branch])

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
        if self.branch_loss_params is not None:
            ek = set(self.branch_loss_params.keys())
            has_same_keys = dk == lk == mk == ek

        if not has_same_keys:
            raise ValueError(
                "Got mismatching keys for branch dict args. "
                f"Branch losses: {lk}. "
                f"Branch loss params: {ek}. "
                f"Decoder branches: {dk}. "
                f"Metrics: {mk}. "
                f"(`metrics`, and `branch_loss_params` can be None)"
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

            loss_kwargs = {"k": None}
            if self.branch_loss_params is not None:
                loss_kwargs = self.branch_loss_params[branch]

            # set joint losses for each branch
            branch_losses[branch] = JointLoss(
                [Loss(loss_name, **loss_kwargs) for loss_name in losses]
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

        if self.optim_params is None:
            self.optim_params = {
                "encoder": {"lr": 0.00005, "weight_decay": 0.00005},
                "decoder": {"lr": 0.0005, "weight_decay": 0.0005},
            }

        params = adjust_optim_params(self.model, self.optim_params)
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
