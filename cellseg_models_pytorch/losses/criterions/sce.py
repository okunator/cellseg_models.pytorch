import torch
import torch.nn.functional as F

from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["SCELoss"]


class SCELoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Symmetric Cross Entropy loss.

        https://arxiv.org/abs/1908.06112

        Args:
            alpha: Weight factor b/w [0,1].
                Defaults to 0.5.
            beta: Weight factor b/w [0,1].
                Defaults to 1.0.
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
        self.beta = beta
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the symmetric cross entropy loss.

        Args:
            yhat: The prediction map. Shape (B, C, H, W).
            target: the ground truth annotations. Shape (B, H, W).
            target_weight: The edge weight map. Shape (B, H, W).
                Defaults to None.
            mask: The mask map. Shape (B, H, W).
                Defaults to None.

        Returns:
            torch.Tensor:
                Computed SCE loss (scalar).
        """
        n_classes = yhat.shape[1]
        target_one_hot = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2)
        yhat_soft = F.softmax(yhat, dim=1) + self.eps
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, n_classes, **kwargs
            )

        forward = target_one_hot * torch.log(yhat_soft)
        reverse = yhat_soft * torch.log(target_one_hot)

        cross_entropy = -torch.sum(forward, dim=1)  # to (B, H, W)
        reverse_cross_entropy = -torch.sum(reverse, dim=1)  # to (B, H, W)
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy

        if self.apply_mask and mask is not None:
            loss = self.apply_mask_weight(loss, mask, norm=False)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
