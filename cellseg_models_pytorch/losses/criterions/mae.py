import torch

from cellseg_models_pytorch.losses.weighted_base_loss import WeightedBaseLoss

__all__ = ["MAE"]


class MAE(WeightedBaseLoss):
    def __init__(
        self,
        alpha: float = 1e-4,
        apply_sd: bool = False,
        apply_mask: bool = False,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        """Compute the MAE loss. Used in the stardist method.

        Stardist:
            https://arxiv.org/pdf/1806.03535.pdf

        Note:
            additionally apply spectral decoupling and edge weights to the loss matrix.

        Parameters:
            alpha (float, default=1e-4)
                Weight regulizer b/w [0,1]. In stardist repo, this is the parameter
                'train_background_reg'.
            apply_sd (bool, default=False):
                If True, applies Spectral decoupling regularization to the loss matrix.
            apply_mask (bool, default=False):
                If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
            edge_weight (float, default=none):
                Weight that is added to object borders.
        """
        super().__init__(apply_sd, False, False, apply_mask, False, edge_weight)
        self.alpha = alpha
        self.eps = 1e-7

    def forward(
        self,
        yhat: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the masked MAE loss.

        Parameters:
            yhat (torch.Tensor):
                The prediction map. Shape (B, C, H, W).
            target (torch.Tensor):
                the ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, default=None):
                The edge weight map. Shape (B, H, W).
            mask (torch.Tensor, default=None):
                The mask map. Shape (B, H, W).

        Returns:
            torch.Tensor:
                Computed MAE loss (scalar).
        """
        n_classes = yhat.shape[1]
        if target.size() != yhat.size():
            target = target.unsqueeze(1).repeat_interleave(n_classes, dim=1)

        if not yhat.shape == target.shape:
            raise ValueError(
                f"Pred and target shapes must match. Got: {yhat.shape}, {target.shape}"
            )

        # compute the MAE loss with alpha as weight
        mae_loss = torch.mean(torch.abs(target - yhat), axis=1)  # (B, H, W)

        if self.apply_mask and mask is not None:
            mae_loss = self.apply_mask_weight(mae_loss, mask, norm=True)  # (B, H, W)

            # add the background regularization
            if self.alpha > 0:
                reg = torch.mean(((1 - mask).unsqueeze(1)) * torch.abs(yhat), axis=1)
                mae_loss += self.alpha * reg

        if self.apply_sd:
            mae_loss = self.apply_spectral_decouple(mae_loss, yhat)

        if self.edge_weight is not None:
            mae_loss = self.apply_edge_weights(mae_loss, target_weight)

        return mae_loss.mean()
