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

    Parameters:
        img1 (torch.Tensor):
            Input image 1. Shape (B, C, H, W).
        img2 (torch.Tensor):
            Input image 2. Shape (B, C, H, W).
        window_size (int, default=11):
            Size of the gaussian kernel.

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
    def __init__(
        self, window_size: int = 11, return_cs: bool = False, **kwargs
    ) -> None:
        """Structural similarity index loss.

        I.e. the dissimilarity:
        (1 - SSIM(x, y)) / 2

        Parameters:
            window_size (int, default=11):
                Size of the gaussian kernel.
            return_cs (bool, default=False):
                Return also the the contrast sensitivity coeff.
        """
        super().__init__()
        self.window_size = window_size
        self.return_cs = return_cs

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the SSIM loss.

        Parameters
            yhat (torch.Tensor):
                The prediction map. Shape (B, C, H, W).
            target (torch.Tensor):
                the ground truth annotations. Shape (B, H, W).
            target_weight (torch.Tensor, default=None):
                The edge weight map. Shape (B, H, W).

        Returns:
            torch.Tensor:
                Computed SSIM loss (scalar).
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
    def __init__(self, window_size: int = 11, **kwargs) -> None:
        """Multi-scale SSIM loss (MS-SSIM).

        UNET3+ https://arxiv.org/pdf/2004.08790.pdf

        penalizes fuzzy boundaries

        Parameters
            window_size (int, default=11):
                Size of the gaussian kernel.
        """
        super().__init__()
        self.window_size = window_size
        self.ssim = SSIM(window_size=window_size, return_cs=True)

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the MS-SSIM loss.

        Parameters:
            yhat (torch.Tensor):
                The prediction map. Shape (B, C, H, W).
            target (torch.Tensor):
                the ground truth annotations. Shape (B, H, W).

        Returns:
            torch.Tensor:
                Computed MS-SSIM loss (scalar).
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
