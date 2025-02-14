import torch
import torch.nn as nn

from cellseg_models_pytorch.utils.convolve import filter2D, gaussian_kernel2d


class WeightedBaseLoss(nn.Module):
    def __init__(
        self,
        apply_sd: bool = False,
        apply_ls: bool = False,
        apply_svls: bool = False,
        apply_mask: bool = False,
        class_weights: torch.Tensor = None,
        edge_weight: float = None,
        **kwargs,
    ) -> None:
        """Init a base class for weighted cross entropy based losses.

        Enables weighting for object instance edges and classes.

        Parameters:
        apply_sd (bool, default=False):
            If True, Spectral decoupling regularization will be applied  to the
            loss matrix.
        apply_ls (bool, default=False):
            If True, Label smoothing will be applied to the target.
        apply_svls (bool, default=False):
            If True, spatially varying label smoothing will be applied to the target
        apply_mask (bool, default=False):
            If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
        class_weights (torch.Tensor, default=None):
            Class weights. A tensor of shape (C, )
        edge_weight (float, default=None):
            Weight for the object instance border pixels
        """
        super().__init__()
        self.apply_sd = apply_sd
        self.apply_ls = apply_ls
        self.apply_svls = apply_svls
        self.apply_mask = apply_mask
        self.class_weights = class_weights
        self.edge_weight = edge_weight

    def apply_spectral_decouple(
        self, loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01
    ) -> torch.Tensor:
        """Apply spectral decoupling L2 norm after the loss.

        https://arxiv.org/abs/2011.09468

        Parameters:
            loss_matrix (torch.Tensor):
                Pixelwise losses. A tensor of shape (B, H, W).
            yhat (torch.Tensor):
                The pixel predictions of the model. Shape (B, C, H, W).
            lam (float, default=0.01):
                Lambda constant.

        Returns:
            torch.Tensor:
                SD-regularized loss matrix. Same shape as input.
        """
        # return loss_matrix + (lam / 2) * (yhat**2).mean() # which??
        return loss_matrix + (lam / 2) * (yhat**2).mean(axis=1)

    def apply_ls_to_target(
        self,
        target: torch.Tensor,
        n_classes: int,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Apply regular label smoothing to the target map.

        https://arxiv.org/abs/1512.00567

        Parameters:
            target (torch.Tensor):
                The target one hot tensor. Shape (B, C, H, W). Dtype: Int64.
            n_classes (int):
                Number of classes in the data.
            label_smoothing (float, default=0.1):
                The smoothing coeff alpha.

        Retrurns:
            Torch.Tensor:
                Label smoothed target. Same shape as input.
        """
        return target * (1 - label_smoothing) + label_smoothing / n_classes

    def apply_svls_to_target(
        self,
        target: torch.Tensor,
        n_classes: int,
        kernel_size: int = 5,
        sigma: int = 3,
        **kwargs,
    ) -> torch.Tensor:
        """Apply spatially varying label smoothihng to target map.

        https://arxiv.org/abs/2104.05788

        Parameters:
            target (torch.Tensor):
                The target one hot tensor. Shape (B, C, H, W). Dtype: Int64.
            n_classes (int):
                Number of classes in the data.
            kernel_size (int, default=3):
                Size of a square kernel.
            sigma (int, default=3):
                The std of the gaussian.

        Retrurns:
            Torch.Tensor:
                Label smoothed target. Same shape as input.
        """
        my, mx = kernel_size // 2, kernel_size // 2
        gaussian_kernel = gaussian_kernel2d(
            kernel_size, sigma, n_classes, device=target.device
        )
        neighborsum = (1 - gaussian_kernel[..., my, mx]) + 1e-16
        gaussian_kernel = gaussian_kernel.clone()
        gaussian_kernel[..., my, mx] = neighborsum
        svls_kernel = gaussian_kernel / neighborsum[0]

        return filter2D(target.float(), svls_kernel) / svls_kernel[0].sum()

    def apply_class_weights(
        self, loss_matrix: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Multiply pixelwise loss matrix by the class weights.

        Note:
            Does not apply normalization

        Parameters:
            loss_matrix (torch.Tensor):
                Pixelwise losses. A tensor of shape (B, H, W).
            target (torch.Tensor):
                The target mask. Shape (B, H, W).

        Returns:
            torch.Tensor:
                The loss matrix scaled with the weight matrix. Shape (B, H, W).
        """
        weight_mat = self.class_weights[target.long()].to(target.device)  # to (B, H, W)
        loss = loss_matrix * weight_mat

        return loss

    def apply_edge_weights(
        self, loss_matrix: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        """Apply weights to the object boundaries.

        Basically just computes `edge_weight`**`weight_map`.

        Parameters:
            loss_matrix (torch.Tensor):
                Pixelwise losses. A tensor of shape (B, H, W).
            weight_map (torch.Tensor):
                Map that points to the pixels that will be weighted.
                Shape (B, H, W).

        Returns:
            torch.Tensor:
                The loss matrix scaled with the nuclear boundary weights.
                Shape (B, H, W).
        """
        return loss_matrix * self.edge_weight**weight_map

    def apply_mask_weight(
        self, loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        """Apply a mask to the loss matrix.

        Parameters:
            loss_matrix (torch.Tensor):
                Pixelwise losses. A tensor of shape (B, H, W).
            mask (torch.Tensor):
                The mask. Shape (B, H, W).
            norm (bool, default=True):
                If True, the loss matrix will be normalized by the mean of the mask.

        Returns:
            torch.Tensor:
                The loss matrix scaled with the mask. Shape (B, H, W).
        """
        loss_matrix *= mask
        if norm:
            norm_mask = torch.mean(mask.float()) + 1e-7
            loss_matrix /= norm_mask

        return loss_matrix

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "apply_sd={apply_sd}, apply_ls={apply_ls}, apply_svls={apply_svls}, apply_mask={apply_mask}, class_weights={class_weights}, edge_weight={edge_weight}"  # noqa
        return s.format(**self.__dict__)
