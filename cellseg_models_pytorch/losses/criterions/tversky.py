import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class TverskyLoss(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
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
            apply_sd : bool, default=False
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
            apply_ls : bool, default=False
                If True, Label smoothing will be applied to the target.
            apply_svls : bool, default=False
                If True, spatially varying label smoothing will be applied to the target
            edge_weight : float, default=none
                Weight that is added to object borders.
            class_weights : torch.Tensor, default=None
                Class weights. A tensor of shape (n_classes,).
        """
        super().__init__(apply_sd, apply_ls, apply_svls, class_weights, edge_weight)
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
        num_classes = yhat.shape[1]
        target_one_hot = tensor_one_hot(target, n_classes=num_classes)
        yhat_soft = F.softmax(yhat, dim=1)
        assert target_one_hot.shape == yhat.shape

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        intersection = torch.sum(yhat_soft * target_one_hot, 1)
        fps = torch.sum(yhat_soft * (1.0 - target_one_hot), 1)
        fns = torch.sum(target_one_hot * (1.0 - yhat_soft), 1)
        denom = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = intersection / denom.clamp_min(self.eps)

        if self.apply_sd:
            tversky_loss = self.apply_spectral_decouple(tversky_loss, yhat)

        if self.class_weights is not None:
            tversky_loss = self.apply_class_weights(tversky_loss, target)

        if self.edge_weight is not None:
            tversky_loss = self.apply_edge_weights(tversky_loss, target_weight)

        return torch.mean(1.0 - tversky_loss)
