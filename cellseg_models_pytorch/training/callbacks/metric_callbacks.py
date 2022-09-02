from typing import Any, Callable, Optional

import torch

from ..functional.train_metrics import accuracy, iou

try:
    from torchmetrics import Metric
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`torchmetrics` package is required when using metric callbacks. "
        "Install with `pip install torchmetrics`"
    )


__all__ = ["Accuracy", "MeanIoU"]


class Accuracy(Metric):
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        progress_group: Any = None,
        dist_sync_func: Callable = None,
    ) -> None:
        """Create a custom torchmetrics accuracy callback.

        Parameters
        ----------
            compute_on_step : bool, default=True
                Forward only calls update() and returns None if this is set to False.
            dist_sync_on_step : bool, default=False
                Synchronize computed values in distributed setting
            process_group : any, optional
                Specify the process group on which synchronization is called.
                default: None (which selects the entire world)
            dist_sync_func : Callable, optional
                Callback that performs the allgather operation on the metric state.
                When None, DDP will be used to perform the allgather.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=progress_group,
            dist_sync_fn=dist_sync_func,
        )

        self.add_state(
            "batch_accuracies", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        activation: str = "softmax",
    ) -> None:
        """Update the batch accuracy list with one batch accuracy value.

        Parameters
        ----------
            pred : torch.Tensor
                Predicted output from the model. Shape (B, C, H, W).
            target : torch.Tensor
                The ground truth segmentation tensor. Shape (B, H, W).
            activation : str, default="softmax"
                The activation function. One of: "softmax", "sigmoid" or None.
        """
        batch_acc = accuracy(pred, target, activation)
        self.batch_accuracies += batch_acc
        self.n_batches += 1

    def compute(self) -> torch.Tensor:
        """Compute the accuracy of one batch and normalize accordingly.

        Returns
        -------
            torch.Tensor: The accuracy value. Shape (1).
        """
        return self.batch_accuracies / self.n_batches


class MeanIoU(Metric):
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        progress_grouo: Any = None,
        dist_sync_func: Callable = None,
    ) -> None:
        """Create a custom torchmetrics mIoU callback.

        Parameters
        ----------
            compute_on_step : bool, default=True
                Forward only calls update() and returns None if this is set to False.
            dist_sync_on_step : bool, default=False
                Synchronize computed values in distributed setting
            process_group : any, optional
                Specify the process group on which synchronization is called.
                default: None (which selects the entire world)
            dist_sync_func : Callable, optional
                Callback that performs the allgather operation on the metric state.
                When None, DDP will be used to perform the allgather.
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=progress_grouo,
            dist_sync_fn=dist_sync_func,
        )

        self.add_state("batch_ious", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        activation: str = "softmax",
    ) -> None:
        """Update the batch IoU list with one batch IoU matrix.

        Parameters
        ----------
            pred : torch.Tensor
                Predicted output from the model. Shape (B, C, H, W).
            target : torch.Tensor
                The ground truth segmentation tensor. Shape (B, H, W).
            activation : str, default="softmax"
                The activation function. One of: "softmax", "sigmoid" or None.
        """
        batch_iou = iou(pred, target, activation)
        self.batch_ious += batch_iou.mean()
        self.n_batches += 1

    def compute(self) -> torch.Tensor:
        """Normalize the batch IoU values.

        Returns
        -------
            torch.Tensor: The IoU mat. Shape (B, n_classes, n_classes).
        """
        return self.batch_ious / self.n_batches


METRIC_LOOKUP = {"acc": Accuracy, "miou": MeanIoU}
