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


class MSSSIM(nn.Module):
    """Combine the package's SSIM dissimilarities across five image scales.

    Supply comparable floating maps on the same device and value scale; these
    losses do not convert raw class scores to probabilities.
    """

    def __init__(self, window_size: int = 11, **kwargs) -> None:
        """Choose the local comparison window.

        Args:
            window_size: Gaussian comparison window width, normally an odd positive integer.
            **kwargs: Additional constructor arguments are accepted but unused.
        """
        super().__init__()
        self.window_size = window_size
        self.ssim = SSIM(window_size=window_size, return_cs=True)

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compare one batch of maps.

        Args:
            yhat: Floating BCHW maps. B is batch size, C channels, H height and W width.
            target: Comparable BCHW floating map. Use spatial dimensions at least 32
                for the five pooling steps; BHW class labels are not supported by this path.
            **kwargs: Additional call arguments are accepted but unused.

        Returns:
            Scalar from the existing weighted five-scale dissimilarity calculation.
            This is not a conventional 1 - MS-SSIM score; identical inputs do not
            necessarily produce zero loss. Results stay on the prediction device with gradients.
        """
        weights = torch.tensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=yhat.device
        )

        levels = weights.size()[0]
        msssim = []
        mcs = []

        for _ in range(levels):
            sim, cs = self.ssim(yhat, target)
            msssim.append(sim)
            mcs.append(cs)

            yhat = F.avg_pool2d(yhat, (2, 2))
            target = F.avg_pool2d(target.float(), (2, 2))

        msssim = torch.stack(msssim)
        mcs = torch.stack(mcs)

        # Normalize to avoid NaNs during training unstable models,
        # not compliant with original definition
        msssim = (msssim + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs**weights
        pow2 = msssim**weights

        loss = torch.prod(pow1[:-1] * pow2[-1])
        return loss
