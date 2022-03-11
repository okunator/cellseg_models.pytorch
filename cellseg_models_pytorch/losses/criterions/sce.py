import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class SCELoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs
    ) -> None:
        """Symmetric Cross Entropy loss.

        https://arxiv.org/abs/1908.06112

        Parameters
        ----------
            alpha : float, default=0.5
                Weight factor b/w [0,1].
            beta :float, default=1.0
                Weight factor b/w [0,1].
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(class_weights, edge_weight)
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute the symmetric cross entropy loss.

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
                Computed SCE loss (scalar).
        """
        num_classes = yhat.shape[1]
        target_one_hot = tensor_one_hot(target, num_classes)
        yhat_soft = F.softmax(yhat, dim=1) + self.eps
        assert target_one_hot.shape == yhat.shape

        yhat = torch.clamp(yhat_soft, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        forward = target_one_hot * torch.log(yhat_soft)
        reverse = yhat_soft * torch.log(target_one_hot)

        cross_entropy = -torch.sum(forward, dim=1)  # to (B, H, W)
        reverse_cross_entropy = -torch.sum(reverse, dim=1)  # to (B, H, W)

        if self.class_weights is not None:
            cross_entropy = self.apply_class_weights(cross_entropy, target)
            reverse_cross_entropy = self.apply_class_weights(
                reverse_cross_entropy, target
            )

        if self.edge_weight is not None:
            cross_entropy = self.apply_edge_weights(cross_entropy, target_weight)
            reverse_cross_entropy = self.apply_edge_weights(
                reverse_cross_entropy, target_weight
            )

        loss = (
            self.alpha * cross_entropy.mean() + self.beta * reverse_cross_entropy.mean()
        )

        return loss
