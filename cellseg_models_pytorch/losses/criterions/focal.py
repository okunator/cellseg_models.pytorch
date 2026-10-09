import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["FocalLoss"]


class FocalLoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Focal loss.

        https://arxiv.org/abs/1708.02002

        Optionally applies, label smoothing, spatially varying label smoothing or
        weights at the object edges or class weights to the loss.

        Args:
            alpha: Weight factor b/w [0,1].
                Defaults to 0.5.
            gamma: Focusing factor.
                Defaults to 2.0.
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
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            yhat: The prediction map. Shape (B, C, H, W).
            target: the ground truth annotations. Shape (B, H, W).
            target_weight: The edge weight map. Shape (B, H, W).
                Defaults to None.
            mask: The mask map. Shape (B, H, W).
                Defaults to None.

        Returns:
            torch.Tensor:
                Computed Focal loss (scalar).
        """
        input_soft = F.softmax(yhat, dim=1) + self.eps  # (B, C, H, W)
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

        weight = (1.0 - input_soft) ** self.gamma
        focal = self.alpha * weight * torch.log(input_soft)
        focal = target_one_hot * focal

        loss = -torch.sum(focal, dim=1)  # to (B, H, W)

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
