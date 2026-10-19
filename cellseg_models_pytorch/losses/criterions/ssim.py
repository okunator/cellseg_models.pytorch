import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.utils.convolve import (
    filter2D,
    gaussian_kernel2d,
)

__all__ = ["SSIM", "MSSSIM"]


def _validate_ssim_options(window_size: int, data_range: float | None) -> None:
    """Reject invalid Gaussian widths and explicit input ranges."""
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 1
    ):
        raise ValueError("window_size must be a positive integer.")
    if data_range is not None and (not math.isfinite(data_range) or data_range <= 0):
        raise ValueError("data_range must be positive and finite.")


def _ssim_filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Return Gaussian-filtered BCHW maps without changing CUDA precision policy."""
    if not image.is_cuda:
        return filter2D(image, kernel)

    height, width = kernel.shape[-2:]
    padding = [(width - 1) // 2, width // 2, (height - 1) // 2, height // 2]
    # ATen exposes per-call precision; the functional API only uses global policy.
    return torch.ops.aten._convolution.default(
        input=F.pad(image, padding, "replicate"),
        weight=kernel,
        bias=None,
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        transposed=False,
        output_padding=[0, 0],
        groups=image.shape[1],
        benchmark=torch.backends.cudnn.benchmark,
        deterministic=(
            torch.backends.cudnn.deterministic
            or torch.are_deterministic_algorithms_enabled()
        ),
        cudnn_enabled=torch.backends.cudnn.enabled,
        allow_tf32=False,
    )


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    val_range: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare floating maps with replicate-padded Gaussian windows.

    Args:
        img1: Floating BCHW maps, without an activation or normalization step.
        img2: Matching floating maps on the same device and value scale.
        window_size: Positive Gaussian width, limited to the smaller spatial axis.
        val_range: Positive finite range, e.g. 1 or 255. None retains legacy
            prediction-only range guessing; use an explicit range for symmetry.

    Returns:
        Raw BCHW similarity and scalar mean contrast similarity, on the input
        device. Half inputs compute in float32; mixed dtypes are promoted and
        float64 is preserved. Both outputs retain gradients.

    Raises:
        ValueError: Shapes differ, inputs are empty or on different devices,
            or the window/range is invalid.
        TypeError: Inputs are not floating maps.
    """
    _validate_ssim_options(window_size, val_range)
    if img1.ndim != 4 or img1.shape != img2.shape or min(img1.shape) < 1:
        raise ValueError("SSIM requires matching nonempty BCHW maps.")
    if img1.device != img2.device:
        raise ValueError("SSIM inputs must share a device.")
    if not img1.is_floating_point() or not img2.is_floating_point():
        raise TypeError("SSIM requires floating maps.")

    dtype = torch.promote_types(img1.dtype, img2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    img1, img2 = img1.to(dtype=dtype), img2.to(dtype=dtype)
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

    with torch.autocast(device_type=img1.device.type, enabled=False):
        (_, channel, height, width) = img1.size()
        real_size = min(window_size, height, width)
        kernel = gaussian_kernel2d(
            real_size, sigma=1.5, n_channels=channel, device=img1.device, dtype=dtype
        )

        # Center maps before squaring to reduce cancellation in local variances.
        offset1 = img1.mean((2, 3), keepdim=True)
        offset2 = img2.mean((2, 3), keepdim=True)
        centered1, centered2 = img1 - offset1, img2 - offset2
        mean1 = _ssim_filter(centered1, kernel)
        mean2 = _ssim_filter(centered2, kernel)
        mu1, mu2 = mean1 + offset1, mean2 + offset2

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _ssim_filter(centered1.square(), kernel) - mean1.square()
        sigma2_sq = _ssim_filter(centered2.square(), kernel) - mean2.square()
        sigma12 = _ssim_filter(centered1 * centered2, kernel) - mean1 * mean2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return ssim_map, cs


class SSIM(nn.Module):
    """Compare local image structure using SSIM dissimilarity.

    Supply comparable floating maps on the same device and value scale; these
    losses do not convert raw class scores to probabilities.
    """

    def __init__(
        self,
        window_size: int = 11,
        return_cs: bool = False,
        data_range: float | None = None,
        **kwargs: object,
    ) -> None:
        """Choose the local comparison window.

        Args:
            window_size: Gaussian comparison window width, normally an odd positive integer.
            return_cs: Also return the single contrast-difference value.
            data_range: Positive finite input range, normally 1 for [0, 1] maps
                or 255 for [0, 255] images. None keeps legacy prediction-only
                range guessing, which can depend on batch contents.
            **kwargs: Additional constructor arguments are accepted but unused.

        Raises:
            ValueError: The window size or explicit data range is invalid.
        """
        super().__init__()
        _validate_ssim_options(window_size, data_range)
        self.window_size = window_size
        self.return_cs = return_cs
        self.data_range = data_range

    def forward(
        self, yhat: torch.Tensor, target: torch.Tensor, **kwargs: object
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compare one batch of maps.

        Args:
            yhat: Floating BCHW maps. B is batch size, C channels, H height and W width.
            target: Matching floating BCHW maps, floating BHW maps shared across
                channels, or integer BHW class IDs in [0, C-1] converted to one-hot
                maps. Inputs must share a device and value scale.
            **kwargs: Additional call arguments are accepted but unused.

        Returns:
            A scalar in [0, 0.5] computed as clamp(1 - mean_similarity, 0, 1) / 2.
            With return_cs=True, returns (loss, contrast_dissimilarity); the latter
            lies in [0, 1]. Results stay on the prediction device with gradients.
            Half inputs compute in float32; mixed dtypes are promoted and float64
            is preserved, including under autocast.

        Raises:
            ValueError: Shapes disagree, inputs are empty or devices differ.
            TypeError: Predictions or BCHW targets are not floating maps.
            RuntimeError: Integer class IDs are outside [0, C-1].

        Examples:
            >>> maps = torch.zeros(1, 1, 8, 8)
            >>> SSIM(data_range=1)(maps, maps).item()
            0.0
        """
        if yhat.ndim != 4 or min(yhat.shape) < 1:
            raise ValueError("SSIM predictions must have nonempty BCHW shape.")
        if not yhat.is_floating_point():
            raise TypeError("SSIM predictions must be floating maps.")
        if target.device != yhat.device:
            raise ValueError("SSIM inputs must share a device.")
        pixel_shape = (yhat.shape[0], *yhat.shape[2:])
        if target.shape == pixel_shape:
            if target.is_floating_point():
                dtype = torch.promote_types(yhat.dtype, target.dtype)
                if dtype in (torch.float16, torch.bfloat16):
                    dtype = torch.float32
                # Sum shared-target gradients before narrowing to its input dtype.
                target = target.to(dtype=dtype).unsqueeze(1).expand_as(yhat)
            elif target.is_complex():
                raise TypeError("SSIM targets must be floating maps or integer labels.")
            else:
                target = F.one_hot(target.long(), yhat.shape[1]).permute(0, 3, 1, 2)
                target = target.to(dtype=yhat.dtype)

        sim, cs = ssim(
            yhat, target, window_size=self.window_size, val_range=self.data_range
        )
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
    # Compare only complete windows, so image borders need no added pixels.
    mu1 = F.conv2d(img1, kernel, groups=channels)
    mu2 = F.conv2d(img2, kernel, groups=channels)
    variance1 = F.conv2d(img1.square(), kernel, groups=channels) - mu1.square()
    variance2 = F.conv2d(img2.square(), kernel, groups=channels) - mu2.square()
    covariance = F.conv2d(img1 * img2, kernel, groups=channels) - mu1 * mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    contrast = (2 * covariance + c2) / (variance1 + variance2 + c2)
    similarity = (2 * mu1 * mu2 + c1) / (mu1.square() + mu2.square() + c1)
    # Keep each image and channel separate until the scales have been combined.
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
        # Four halvings must still leave a full comparison window at the fifth scale.
        minimum_size = max(16, 16 * (self.window_size - 1) + 1)
        if min(yhat.shape[:2]) < 1 or min(yhat.shape[2:]) < minimum_size:
            raise ValueError(
                f"MSSSIM requires nonempty batches/channels and spatial axes "
                f"of at least {minimum_size} pixels for window_size={self.window_size}."
            )

        # Autocast could otherwise lower convolution precision despite the input dtype.
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
            # Standard MS-SSIM weights set how much each image scale contributes.
            weights = yhat.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
            components = []
            for level in range(5):
                similarity, contrast = _ssim_per_channel(
                    yhat, target, kernel, self.data_range
                )
                # Compare contrast at every scale, and brightness only at the last.
                components.append(contrast if level < 4 else similarity)
                if level < 4:
                    # Pad odd image sizes so both maps shrink in the same way.
                    padding = (yhat.shape[2] % 2, yhat.shape[3] % 2)
                    yhat = F.avg_pool2d(yhat, 2, padding=padding)
                    target = F.avg_pool2d(target, 2, padding=padding)

            # Negative scores cannot be raised to fractional powers safely.
            scores = torch.stack(components).relu().clamp_max(1)
            # Combine scales per image/channel before averaging the batch.
            return 1 - (scores ** weights[:, None, None]).prod(0).mean()
