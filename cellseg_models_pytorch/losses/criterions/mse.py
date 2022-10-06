import torch
import torch.nn.functional as F

from cellseg_models_pytorch.utils import tensor_one_hot

from ..weighted_base_loss import WeightedBaseLoss


class MSE(WeightedBaseLoss):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        edge_weight: float = None,
        class_weights: torch.Tensor = None,
        **kwargs,
    ) -> None:
        """MSE-loss.

        Parameters
        ----------
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

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the MSE-loss.

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
                Computed MSE loss (scalar).
        """
        target_one_hot = target
        num_classes = yhat.shape[1]

        if target.size() != yhat.size():
            if target.dtype == torch.float32:
                target_one_hot = target.unsqueeze(1)
            else:
                target_one_hot = tensor_one_hot(target, num_classes)

        if self.apply_svls:
            target_one_hot = self.apply_svls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        if self.apply_ls:
            target_one_hot = self.apply_ls_to_target(
                target_one_hot, num_classes, **kwargs
            )

        mse = F.mse_loss(yhat, target_one_hot, reduction="none")  # (B, C, H, W)
        mse = torch.mean(mse, dim=1)  # to (B, H, W)

        if self.apply_sd:
            mse = self.apply_spectral_decouple(mse, yhat)

        if self.class_weights is not None:
            mse = self.apply_class_weights(mse, target)

        if self.edge_weight is not None:
            mse = self.apply_edge_weights(mse, target_weight)

        return torch.mean(mse)
