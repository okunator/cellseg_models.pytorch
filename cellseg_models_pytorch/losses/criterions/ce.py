import torch
import torch.nn.functional as F

from ...utils import tensor_one_hot
from ..weighted_base_loss import WeightedBaseLoss

__all__ = ["CELoss"]


class CELoss(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """Cross-Entropy loss with weighting.

        Parameters
        ----------
            apply_sd : bool, default=False
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
            apply_ls : bool, default=False
                If True, Label smoothing will be applied to the target.
            apply_svls : bool, default=False
                If True, spatially varying label smoothing will be applied to the target
            edge_weight : float, default=None
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(apply_sd, apply_ls, apply_svls, class_weights, edge_weight)
        self.eps = 1e-8

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the cross entropy loss.

        Parameters
        ----------
            yhat : torch.Tensor
                The prediction map. Shape (B, C, H, W).
            target : torch.Tensor
                the ground truth annotations. Shape (B, H, W).
            target_weight : torch.Tensor, default=None
                The edge weight map. Shape (B, H, W).

        Returns
        -------
            torch.Tensor:
                Computed CE loss (scalar).
        """
        input_soft = F.softmax(yhat, dim=1) + self.eps  # (B, C, H, W)
        num_classes = yhat.shape[1]
        target_one_hot = tensor_one_hot(target, num_classes)  # (B, C, H, W)
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        loss = -torch.sum(target_one_hot * torch.log(input_soft), dim=1)  # (B, H, W)

        if self.apply_sd:
            loss = self.apply_spectral_decouple(loss, yhat)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
