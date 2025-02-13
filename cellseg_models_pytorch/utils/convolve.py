import torch
import torch.nn.functional as F

__all__ = ["filter2D", "gaussian", "gaussian_kernel2d"]


def filter2D(input_tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolves a given kernel on input tensor without losing dimensional shape.

    Parameters
    ----------
        input_tensor : torch.Tensor
            Input image/tensor.
        kernel : torch.Tensor
            Convolution kernel/window.

    Returns
    -------
        torch.Tensor:
            The convolved tensor of same shape as the input.
    """
    (_, channel, _, _) = input_tensor.size()

    # "SAME" padding to avoid losing height and width
    pad = [
        kernel.size(2) // 2,
        kernel.size(2) // 2,
        kernel.size(3) // 2,
        kernel.size(3) // 2,
    ]
    pad_tensor = F.pad(input_tensor, pad, "replicate")

    out = F.conv2d(pad_tensor, kernel, groups=channel)
    return out


def gaussian(
    window_size: int, sigma: float, device: torch.device = None
) -> torch.Tensor:
    """Create a gaussian 1D tensor.

    Parameters
    ----------
        window_size : int
            Number of elements for the output tensor.
        sigma : float
            Std of the gaussian distribution.
        device : torch.device
            Device for the tensor.

    Returns
    -------
        torch.Tensor:
            A gaussian 1D tensor. Shape: (window_size, ).
    """
    x = torch.arange(window_size, device=device).float() - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp((-x.pow(2.0) / float(2 * sigma**2)))

    return gauss / gauss.sum()


def gaussian_kernel2d(
    window_size: int, sigma: float, n_channels: int = 1, device: torch.device = None
) -> torch.Tensor:
    """Create 2D window_size**2 sized kernel a gaussial kernel.

    Parameters
    ----------
        window_size : int
            Number of rows and columns for the output tensor.
        sigma : float
            Std of the gaussian distribution.
        n_channel : int
            Number of channels in the image that will be convolved with
            this kernel.
        device : torch.device
            Device for the kernel.

    Returns:
    -----------
        torch.Tensor:
            A tensor of shape (1, 1, window_size, window_size)
    """
    kernel_x = gaussian(window_size, sigma, device=device)
    kernel_y = gaussian(window_size, sigma, device=device)

    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    kernel_2d = kernel_2d.expand(n_channels, 1, window_size, window_size)

    return kernel_2d
