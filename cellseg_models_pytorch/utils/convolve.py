# Portions adapted from Kornia (Apache-2.0) for this library.
# See NOTICE.md for attribution and license text.

from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F

__all__ = ["filter2D", "gaussian", "gaussian_kernel2d"]


# https://github.com/kornia/kornia/blob/main/kornia/filters/filter.py#L32
def _compute_padding(kernel_size: Sequence[int]) -> list[int]:
    """Compute padding tuple."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def filter2D(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolves a given kernel on input tensor without losing dimensional shape.

    Args:
        input_tensor: Floating BCHW image tensor.
        kernel: Kernel of shape (C, 1, KH, KW), on the input device and dtype.

    Returns:
        The convolved BCHW tensor, preserving shape, dtype and device.
        Replicate padding includes asymmetric padding for even kernels.
    """
    (_, channel, _, _) = input_tensor.size()

    # "SAME" padding to avoid losing height and width
    pad = _compute_padding(kernel.shape[2:])
    pad_tensor = F.pad(input_tensor, pad, "replicate")
    out = F.conv2d(pad_tensor, kernel, groups=channel)
    return out


def gaussian(
    window_size: int,
    sigma: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a gaussian 1D tensor.

    Args:
        window_size: Number of elements for the output tensor.
        sigma: Std of the gaussian distribution.
        device: Device for the tensor.
        dtype: Data type for the tensor.

    Returns:
        A normalized tensor of shape (window_size,), on the requested device.
        Defaults to float32 and PyTorch's default device.
    """
    if dtype is None:
        dtype = torch.float32

    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5

    # Meta tensors need no precision guard; their device does not support autocast.
    context = (
        torch.autocast(x.device.type, enabled=False)
        if torch.amp.autocast_mode.is_autocast_available(x.device.type)
        else nullcontext()
    )
    with context:
        gauss = torch.exp((-x.pow(2.0) / float(2 * sigma**2)))
        return gauss / gauss.sum()


def gaussian_kernel2d(
    window_size: int,
    sigma: float,
    n_channels: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Create a normalized square Gaussian kernel for depthwise convolution.

    Args:
        window_size: Number of rows and columns for the output tensor.
        sigma: Std of the gaussian distribution.
        n_channels: Number of channels in the image to convolve.
        device: Device for the kernel.
        dtype: Data type for the kernel.

    Returns:
        A tensor of shape (n_channels, 1, window_size, window_size),
        with each channel summing to one. Defaults to float32 and PyTorch's
        default device.

    Examples:
        >>> kernel = gaussian_kernel2d(3, 1.0, n_channels=2)
        >>> kernel.shape
        torch.Size([2, 1, 3, 3])
    """
    kernel_x = gaussian(window_size, sigma, device=device, dtype=dtype)
    kernel_y = gaussian(window_size, sigma, device=device, dtype=dtype)

    # Elementwise products keep the requested dtype inside autocast.
    kernel_2d = kernel_x[:, None] * kernel_y[None, :]
    kernel_2d = kernel_2d.expand(n_channels, 1, window_size, window_size)

    return kernel_2d
