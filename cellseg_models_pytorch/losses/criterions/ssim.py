import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.utils.convolve import (
    filter2D,
    gaussian_kernel2d,
)

__all__ = ["SSIM", "MSSSIM"]


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    val_range: int = None,
) -> torch.Tensor:
    """Structural similarity index (SSIM).

    UNET3+ https://arxiv.org/pdf/2004.08790.pdf

    Args:
        img1: Input image 1. Shape (B, C, H, W).
        img2: Input image 2. Shape (B, C, H, W).
        window_size: Size of the gaussian kernel.
            Defaults to 11.

    Returns:
        torch.Tensor:
            Computed ssim loss and contrast sensitivity.
    """
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    (_, channel, height, width) = img1.size()
    real_size = min(window_size, height, width)
    kernel = gaussian_kernel2d(
        real_size, sigma=1.5, n_channels=channel, device=img1.device
    )

    mu1 = filter2D(img1, kernel)
    mu2 = filter2D(img2, kernel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2D(img1 * img1, kernel) - mu1_sq
    sigma2_sq = filter2D(img2 * img2, kernel) - mu2_sq
    sigma12 = filter2D(img1 * img2, kernel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return ssim_map, cs


class SSIM(nn.Module):
    """Compare local image structure using SSIM dissimilarity.

    Supply comparable floating maps on the same device and value scale; these
    losses do not convert raw class scores to probabilities.
    """

    def __init__(
        self, window_size: int = 11, return_cs: bool = False, **kwargs
    ) -> None:
        """Choose the local comparison window.

        Args:
            window_size: Gaussian comparison window width, normally an odd positive integer.
            return_cs: Also return the single contrast-difference value.
            **kwargs: Additional constructor arguments are accepted but unused.
        """
        super().__init__()
        self.window_size = window_size
        self.return_cs = return_cs

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compare one batch of maps.

        Args:
            yhat: Floating BCHW maps. B is batch size, C channels, H height and W width.
            target: Comparable BCHW floating map. BHW labels are one-hot encoded when
                possible; otherwise a channel axis is added.
            **kwargs: Additional call arguments are accepted but unused.

        Returns:
            A scalar in [0, 0.5] computed as clamp(1 - mean_similarity, 0, 1) / 2.
            With return_cs=True, returns (loss, contrast_dissimilarity); the latter
            lies in [0, 1]. Results stay on the prediction device with gradients.
        """
        # if target is label mask shaped (B, H, W) then one hot
        if len(target.shape) == 3:
            try:
                n_classes = yhat.shape[1]
                target = F.one_hot(target.long(), n_classes).permute(0, 3, 1, 2).float()
            except Exception:
                target = target.unsqueeze(1)

        # compute ssim
        sim, cs = ssim(yhat, target, window_size=self.window_size)
        loss = torch.clamp(1.0 - sim.mean(), min=0, max=1) / 2.0

        if self.return_cs:
            cs = torch.clamp(1.0 - cs, min=0, max=1)
            loss = (loss, cs)

        return loss


def _ssim_per_channel(
    img1: torch.Tensor,
    img2: torch.Tensor,
    kernel: torch.Tensor,
    data_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw SSIM and contrast similarities per image/channel.

    Inputs are matching floating BCHW maps and a same-device/dtype Gaussian
    kernel. Valid convolution excludes image borders; both outputs have shape BC.
    """
    channels = img1.shape[1]
    mu1 = F.conv2d(img1, kernel, groups=channels)
    mu2 = F.conv2d(img2, kernel, groups=channels)
    variance1 = F.conv2d(img1.square(), kernel, groups=channels) - mu1.square()
    variance2 = F.conv2d(img2.square(), kernel, groups=channels) - mu2.square()
    covariance = F.conv2d(img1 * img2, kernel, groups=channels) - mu1 * mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    contrast = (2 * covariance + c2) / (variance1 + variance2 + c2)
    similarity = (2 * mu1 * mu2 + c1) / (mu1.square() + mu2.square() + c1)
    return (similarity * contrast).mean((2, 3)), contrast.mean((2, 3))


class MSSSIM(nn.Module):
    """Compare floating maps using 1 minus five-scale MS-SSIM.

    Contrast is compared at each scale; brightness is compared only at the
    coarsest scale. Image/channel scores are combined before averaging.
    Inputs are not normalized or converted from logits or class labels.
    """

    def __init__(
        self, window_size: int = 11, data_range: float = 1.0, **kwargs: object
    ) -> None:
        """Choose the Gaussian window and fixed input range.

        Args:
            window_size: Positive odd Gaussian window width; sigma is 1.5.
            data_range: Positive finite input range, normally 1 for maps in [0, 1]
                or 255 for floating images in [0, 255]. Defaults to 1.
            **kwargs: Accepted but unused, including weighting and smoothing options.

        Raises:
            ValueError: The window size or data range is invalid.
        """
        super().__init__()
        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size < 1
            or window_size % 2 == 0
        ):
            raise ValueError("window_size must be a positive odd integer.")
        if not math.isfinite(data_range) or data_range <= 0:
            raise ValueError("data_range must be positive and finite.")
        self.window_size = window_size
        self.data_range = data_range

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        """Compare one batch of maps on their existing device.

        Args:
            yhat: Floating BCHW maps in [0, data_range]. B is batch size, C
                channels, H height and W width. Each spatial axis must be at least
                max(16, 16 * (window_size - 1) + 1): 161 for the default window.
            target: Matching BCHW maps with the same dtype, device and value range.
            **kwargs: Accepted but unused, including masks and pixel weights.

        Returns:
            Scalar loss in [0, 1], zero for identical maps, with gradients on the
            input device. Float16/bfloat16 compute in float32, including under
            autocast; float32/float64 retain their dtype.

        Raises:
            ValueError: Input shapes, devices or dtypes differ, a batch/channel
                axis is empty, or spatial axes are too small.
            TypeError: Inputs are not floating tensors.

        Examples:
            >>> maps = torch.zeros(1, 1, 161, 161)
            >>> MSSSIM()(maps, maps).item()
            0.0
        """
        if yhat.ndim != 4 or yhat.shape != target.shape:
            raise ValueError("MSSSIM requires matching BCHW maps.")
        if yhat.device != target.device or yhat.dtype != target.dtype:
            raise ValueError("MSSSIM inputs must have the same device and dtype.")
        if not yhat.is_floating_point():
            raise TypeError("MSSSIM requires floating maps, not class labels.")
        minimum_size = max(16, 16 * (self.window_size - 1) + 1)
        if min(yhat.shape[:2]) < 1 or min(yhat.shape[2:]) < minimum_size:
            raise ValueError(
                f"MSSSIM requires nonempty batches/channels and spatial axes "
                f"of at least {minimum_size} pixels for window_size={self.window_size}."
            )

        # Convolution must also avoid low-precision autocast, not just its inputs.
        with torch.autocast(device_type=yhat.device.type, enabled=False):
            if yhat.dtype in (torch.float16, torch.bfloat16):
                yhat, target = yhat.float(), target.float()
            kernel = gaussian_kernel2d(
                self.window_size,
                sigma=1.5,
                n_channels=yhat.shape[1],
                device=yhat.device,
                dtype=yhat.dtype,
            )
            weights = yhat.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
            components = []
            for level in range(5):
                similarity, contrast = _ssim_per_channel(
                    yhat, target, kernel, self.data_range
                )
                components.append(contrast if level < 4 else similarity)
                if level < 4:
                    # Symmetric zero padding preserves odd axes when downsampling.
                    padding = (yhat.shape[2] % 2, yhat.shape[3] % 2)
                    yhat = F.avg_pool2d(yhat, 2, padding=padding)
                    target = F.avg_pool2d(target, 2, padding=padding)

            # ReLU keeps negative similarities safe under fractional powers.
            scores = torch.stack(components).relu().clamp_max(1)
            return 1 - (scores ** weights[:, None, None]).prod(0).mean()
