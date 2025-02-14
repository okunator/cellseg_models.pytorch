import torch
import torch.nn.functional as F

from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss

__all__ = ["BCELoss"]


class BCELoss(WeightedBaseLoss):
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
        """Binary cross entropy loss with weighting and other tricks.

        Parameters
            apply_sd (bool, default=False):
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
            apply_ls (bool, default=False):
                If True, Label smoothing will be applied to the target.
            apply_svls (bool, default=False):
                If True, spatially varying label smoothing will be applied to the target
            apply_mask (bool, default=False):
                If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
            edge_weight (float, default=None):
                Weight that is added to object borders.
            class_weights (torch.Tensor, default=None):
                Class weights. A tensor of shape (n_classes,).
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
        """Compute binary cross entropy loss.

        Parameters:
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
                Computed BCE loss (scalar).
        """
        num_classes = yhat.shape[1]
        yhat = torch.clip(yhat, self.eps, 1.0 - self.eps)

        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(num_classes, dim=1)

        if self.apply_svls:
            target = self.apply_svls_to_target(target, num_classes, **kwargs)

        if self.apply_ls:
            target = self.apply_ls_to_target(target, num_classes, **kwargs)

        bce = F.binary_cross_entropy_with_logits(
            yhat.float(), target.float(), reduction="none"
        )  # (B, C, H, W)
        bce = torch.mean(bce, dim=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            bce = self.apply_mask_weight(bce, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            bce = self.apply_spectral_decouple(bce, yhat)

        if self.class_weights is not None:
            bce = self.apply_class_weights(bce, target)

        if self.edge_weight is not None:
            bce = self.apply_edge_weights(bce, target_weight)

        return torch.mean(bce)
