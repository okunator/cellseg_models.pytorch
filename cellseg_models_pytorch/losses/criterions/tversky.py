import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class TverskyLoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs
    ) -> None:
        """Tversky loss.

        https://arxiv.org/abs/1706.05721

        Parameters
        ----------
            alpha : float, default=0.7
                 False positive dice coefficient.
            beta : float, default=0.3
                False negative tanimoto coefficient.
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(class_weights, edge_weight)
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute the Tversky loss.

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
                Computed Tversky loss (scalar).
        """
        target_one_hot = tensor_one_hot(target, n_classes=yhat.shape[1])
        yhat_soft = F.softmax(yhat, dim=1)
        assert target_one_hot.shape == yhat.shape

        intersection = torch.sum(yhat_soft * target_one_hot, 1)
        fps = torch.sum(yhat_soft * (1.0 - target_one_hot), 1)
        fns = torch.sum(target_one_hot * (1.0 - yhat_soft), 1)
        denom = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = intersection / denom.clamp_min(self.eps)

        if self.class_weights is not None:
            tversky_loss = self.apply_class_weights(tversky_loss, target)

        if self.edge_weight is not None:
            tversky_loss = self.apply_edge_weights(tversky_loss, target_weight)

        return torch.mean(1.0 - tversky_loss)
