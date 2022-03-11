import torch
import torch.nn as nn

from ..weighted_base_loss import WeightedBaseLoss


class CELoss(WeightedBaseLoss):
    def __init__(
        self, edge_weight: float = None, class_weights: torch.Tensor = None, **kwargs
    ) -> None:
        """Cross-Entropy loss with weighting.

        Parameters
        ----------
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(class_weights, edge_weight)
        self.loss = nn.CrossEntropyLoss(reduction="none", weight=class_weights)

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs
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
        loss = self.loss(yhat, target)  # (B, H, W)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
