import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class DiceLoss(WeightedBaseLoss):
    def __init__(
        self, edge_weight: float = None, class_weights: torch.Tensor = None, **kwargs
    ) -> None:
        """Sørensen-Dice Coefficient Loss.

        Optionally applies weights at the object edges and classes.

        Parameters
        ----------
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(class_weights, edge_weight)
        self.eps = 1e-6

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute the DICE coefficient.

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
                Computed DICE loss (scalar).
        """
        yhat_soft = F.softmax(yhat, dim=1)
        target_one_hot = tensor_one_hot(target, n_classes=yhat.shape[1])
        assert target_one_hot.shape == yhat.shape

        intersection = torch.sum(yhat_soft * target_one_hot, 1)
        union = torch.sum(yhat_soft + target_one_hot, 1)
        dice = 2.0 * intersection / union.clamp_min(self.eps)

        if self.class_weights is not None:
            dice = self.apply_class_weights(dice, target)

        if self.edge_weight is not None:
            dice = self.apply_edge_weights(dice, target_weight)

        return torch.mean(1.0 - dice)
