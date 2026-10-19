import torch
import torch.nn as nn

from cellseg_models_pytorch.utils.convolve import filter2D


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

        Args:
            apply_sd:
                If True, Spectral decoupling regularization will be applied  to the
                loss matrix.
                Defaults to False.
            apply_ls: If True, Label smoothing will be applied to the target.
                Defaults to False.
            apply_svls:
                If True, spatially varying label smoothing will be applied to the target
                Defaults to False.
            apply_mask:
                If True, a mask will be applied to the loss matrix. Mask shape: (B, H, W)
                Defaults to False.
            class_weights: Class weights. A tensor of shape (C, )
                Defaults to None.
            edge_weight: Weight for the object instance border pixels
                Defaults to None.
        """
        super().__init__()
        self.apply_sd = apply_sd
        self.apply_ls = apply_ls
        self.apply_svls = apply_svls
        self.apply_mask = apply_mask
        # Move class weights with the loss module without adding them to checkpoints.
        self.register_buffer("class_weights", class_weights, persistent=False)
        self.edge_weight = edge_weight

    def apply_spectral_decouple(
        self, loss_matrix: torch.Tensor, yhat: torch.Tensor, lam: float = 0.01
    ) -> torch.Tensor:
        """Apply spectral decoupling L2 norm after the loss.

        https://arxiv.org/abs/2011.09468

        Args:
            loss_matrix: Pixelwise losses. A tensor of shape (B, H, W).
            yhat: The pixel predictions of the model. Shape (B, C, H, W).
            lam: Lambda constant.
                Defaults to 0.01.

        Returns:
            torch.Tensor:
                SD-regularized loss matrix. Same shape as input.
        """
        # Keep one penalty per pixel so masks and weights can still be applied.
        return loss_matrix + (lam / 2) * (yhat**2).mean(axis=1)

    def apply_ls_to_target(
        self,
        target: torch.Tensor,
        n_classes: int,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """Apply regular label smoothing to the target map.

        https://arxiv.org/abs/1512.00567

            target (torch.Tensor):
                The target one hot tensor. Shape (B, C, H, W). Dtype: Int64.
            n_classes (int):
                Number of classes in the data.
            label_smoothing (float, default=0.1):
                The smoothing coeff alpha.


        Args:
            Retrurns: Torch.Tensor:
                    Label smoothed target. Same shape as input.
        """
        return target * (1 - label_smoothing) + label_smoothing / n_classes

    def apply_svls_to_target(
        self,
        target: torch.Tensor,
        n_classes: int,
        kernel_size: int = 5,
        sigma: float = 3.0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Smooth targets using their neighboring pixels.

        See https://arxiv.org/abs/2104.05788 for spatially varying label smoothing.

        Args:
            target: BCHW class maps, normally one-hot encoded labels.
            n_classes: Number of channels C.
            kernel_size: Square Gaussian smoothing window width.
            sigma: Gaussian standard deviation.
            **kwargs: Accepted but unused, including uniform smoothing options.

        Returns:
            Smoothed BCHW targets on the input device, in float32 even under
            autocast. The center and its neighbors each supply half the weight
            when neighbors exist. A one-pixel window returns the input in float32.
        """
        # Build targets in float32 so mixed precision does not change the labels.
        with torch.autocast(device_type=target.device.type, enabled=False):
            if kernel_size == 1:
                return target.float()
            my, mx = kernel_size // 2, kernel_size // 2
            offsets = (
                torch.arange(kernel_size, device=target.device, dtype=torch.float32)
                - (kernel_size - 1) / 2
            )
            log_weights = -(offsets.square()[:, None] + offsets.square()[None, :]) / (
                2 * sigma**2
            )
            # Normalize neighbors separately so narrow Gaussians do not lose their weight.
            log_weights[my, mx] = -torch.inf
            neighbors = (
                log_weights.flatten().softmax(0).reshape(kernel_size, kernel_size)
            )
            # The center and all neighbors together each contribute half the target.
            neighbors[my, mx] = 1
            svls_kernel = (
                neighbors[None, None].expand(n_classes, 1, kernel_size, kernel_size) / 2
            )

            return filter2D(target.float(), svls_kernel)

    def apply_class_weights(
        self, loss_matrix: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Multiply pixelwise loss matrix by the class weights.

        Note:
            Does not apply normalization

        Args:
            loss_matrix: Pixelwise losses. A tensor of shape (B, H, W).
            target: The target mask. Shape (B, H, W).

        Returns:
            torch.Tensor:
                The loss matrix scaled with the weight matrix. Shape (B, H, W).
        """
        # Keep weights tied to original class IDs even when loss targets were smoothed.
        weight_mat = self.class_weights[target.long()].to(target.device)  # to (B, H, W)
        loss = loss_matrix * weight_mat

        return loss

    def apply_edge_weights(
        self, loss_matrix: torch.Tensor, weight_map: torch.Tensor
    ) -> torch.Tensor:
        """Scale pixel losses by edge_weight raised to each edge exponent.

        Args:
            loss_matrix: Floating BHW or BCHW pixel losses.
            weight_map: Edge exponents broadcastable to the loss shape, on its
                device. Powers use the loss dtype to avoid half-precision overflow.

        Returns:
            Weighted pixel losses with the input shape, on the input device.
        """
        # Widen exponents before taking powers to avoid float16 overflow.
        return loss_matrix * self.edge_weight ** weight_map.to(dtype=loss_matrix.dtype)

    def apply_mask_weight(
        self, loss_matrix: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        """Apply a mask to the loss matrix.

        Args:
            loss_matrix: Pixelwise losses. A tensor of shape (B, H, W).
            mask: The mask. Shape (B, H, W).
            norm:
                If True, the loss matrix will be normalized by the mean of the mask.
                Defaults to True.

        Returns:
            torch.Tensor:
                The loss matrix scaled with the mask. Shape (B, H, W).
        """
        loss_matrix *= mask
        if norm:
            # Remove the mask's effect on the average, leaving a weighted pixel mean.
            norm_mask = mask.to(dtype=loss_matrix.dtype).mean() + 1e-7
            loss_matrix /= norm_mask

        return loss_matrix

    def extra_repr(self) -> str:
        """Add info to print."""
        s = "apply_sd={apply_sd}, apply_ls={apply_ls}, apply_svls={apply_svls}, apply_mask={apply_mask}, class_weights={class_weights}, edge_weight={edge_weight}"  # noqa
        return s.format(**(self.__dict__ | {"class_weights": self.class_weights}))
