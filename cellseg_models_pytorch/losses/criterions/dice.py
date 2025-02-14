import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["DiceLoss"]


class DiceLoss(WeightedBaseLoss):
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
        """Sørensen-Dice Coefficient Loss.

        Optionally applies weights at the object edges and classes.

        Parameters
        ----------
        apply_sd : bool, default=False
            If True, Spectral decoupling regularization will be applied  to the
            loss matrix.
        apply_ls : bool, default=False
            If True, Label smoothing will be applied to the target.
        apply_svls : bool, default=False
            If True, spatially varying label smoothing will be applied to the target
        apply_mask : bool, default=False
            If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
        edge_weight : float, default=none
            Weight that is added to object borders.
        class_weights : torch.Tensor, default=None
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
        """Compute the DICE coefficient.

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
                Computed DICE loss (scalar).
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

        intersection = torch.sum(yhat_soft * target_one_hot, 1)
        union = torch.sum(yhat_soft + target_one_hot, 1)
        dice = 2.0 * intersection / union.clamp_min(self.eps)  # (B, H, W)

        if self.apply_mask and mask is not None:
            dice = self.apply_mask_weight(dice, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            dice = self.apply_spectral_decouple(dice, yhat)

        if self.class_weights is not None:
            dice = self.apply_class_weights(dice, target)

        if self.edge_weight is not None:
            dice = self.apply_edge_weights(dice, target_weight)

        return torch.mean(1.0 - dice)
