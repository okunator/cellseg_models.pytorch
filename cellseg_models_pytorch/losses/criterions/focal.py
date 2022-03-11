import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class FocalLoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs
    ) -> None:
        """Focal loss.

        https://arxiv.org/abs/1708.02002

        Optionally applies weights at the object edges and classes.

        Parameters
        ----------
            alpha : float, default=0.5
                Weight factor b/w [0,1].
            gamma : float, default=2.0
                Focusing factor.
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(class_weights, edge_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute the focal loss.

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
                Computed Focal loss (scalar).
        """
        input_soft = F.softmax(yhat, dim=1) + self.eps  # (B, C, H, W)
        num_classes = yhat.shape[1]
        target_one_hot = tensor_one_hot(target, num_classes)  # (B, C, H, W)
        assert target_one_hot.shape == yhat.shape

        weight = (1.0 - input_soft) ** self.gamma
        focal = self.alpha * weight * torch.log(input_soft)
        focal = target_one_hot * focal

        loss = -torch.sum(focal, dim=1)  # to (B, H, W)

        if self.class_weights is not None:
            loss = self.apply_class_weights(loss, target)

        if self.edge_weight is not None:
            loss = self.apply_edge_weights(loss, target_weight)

        return loss.mean()
