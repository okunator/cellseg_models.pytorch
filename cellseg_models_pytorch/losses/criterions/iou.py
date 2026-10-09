import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["IoULoss"]


class IoULoss(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Intersection over union loss.

        Optionally applies weights at the object edges and classes.

        Args:
            apply_sd:
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
                Defaults to False.
            apply_ls: If True, Label smoothing will be applied to the target.
                Defaults to False.
            apply_svls:
                If True, spatially varying label smoothing will be applied to the target
                Defaults to False.
            apply_mask:
                If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
                Defaults to False.
            edge_weight: Weight that is added to object borders.
                Defaults to none.
            class_weights: Class weights. A tensor of shape (n_classes,).
                Defaults to None.
        """
        super().__init__(
            apply_sd, apply_ls, apply_svls, apply_mask, class_weights, edge_weight
        )
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the IoU loss.

        Parameters
            yhat (torch.Tensor):
                The prediction map. Shape (B, C, H, W).
            target (torch.Tensor):
                the ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, default=None):
                The edge weight map. Shape (B, H, W).
            mask (torch.Tensor, default=None):
                The mask map. Shape (B, H, W).

        Returns:
            torch.Tensor:
                Computed IoU loss (scalar).
        """
        yhat_soft = F.softmax(yhat, dim=1)
        n_classes = yhat.shape[1]
        target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)

        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        intersection = torch.sum(yhat_soft * target_one_hot, 1)  # to (B, H, W)
        union = torch.sum(yhat_soft + target_one_hot, 1)  # to (B, H, W)
        iou = intersection / union.clamp_min(self.eps)

        if self.apply_mask and mask is not None:
            iou = self.apply_mask_weight(iou, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            iou = self.apply_spectral_decouple(iou, yhat)

        if self.class_weights is not None:
            iou = self.apply_class_weights(iou, target)

        if self.edge_weight is not None:
            iou = self.apply_edge_weights(iou, target_weight)

        return torch.mean(1.0 - iou)
