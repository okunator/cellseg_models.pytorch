from typing import Tuple, Union

import torch

__all__ = [
    "dataset_normalize_torch",
    "percentile",
    "percentile_normalize_torch",
    "minmax_normalize_torch",
    "normalize_torch",
    "NORM_LOOKUP",
]


def dataset_normalize_torch(
    img: torch.Tensor,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    to_uint8: bool = True,
) -> torch.Tensor:
    """Normalize a 3-channel img tensor with mean and standard deviation of the dataset.

    Parameters
    ----------
        img : torch.Tensor
            Tensor img of Shape (C, H, W) or (B, C, H, W).
        mean : Tuple[float, float, float]
            Means for each channel.
        std : Tuple[float, float, float]
            Standard deviations for each channel.
        to_uint8 : bool, default=True
            If input tensor values between [0, 255]. The std and mean
            can be scaled from [0, 1] -> [0, 255].

    Raises
    ------
        TypeError if input img is not torch.Tensor.
        ValueError if input image has illegal shape.
        ZeroDivisionError if normalizing leads to zero-div error.

    Returns
    -------
        torch.Tensor:
            Normalized Tensor image. Same shape as input.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"input img needs to be a tensor. Got {img.dtype}.")

    if not 3 <= img.ndim <= 4:
        raise ValueError("img tensor shape should be either (C, H, W)|(B, C, H, W)")

    if len(mean) != 3 and len(std) != 3:
        raise ValueError(
            f"Mean and std is needed for every channel. Got: {mean} and {std}"
        )

    img = img.float()
    mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
    std = torch.tensor(std, dtype=img.dtype, device=img.device)

    if to_uint8:
        mean = mean * 255
        std = std * 255

    if (std == 0).any():
        raise ZeroDivisionError(
            "zeros detected in std-values -> would lead to zero-div error"
        )

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    img.sub_(mean).div_(std)

    return img


# Channel-wise normalizations per image.
# Ported from: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100
              inclusive.

    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted
    # value indeed corresponds to k=1, not k=0! Use float(q) instead of
    # q directly, so that ``round()`` returns an integer, even if q is
    # a np.float32.
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.contiguous().view(-1).kthvalue(k).values.item()
    return result


def percentile_normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """1-99 percentile normalization per image channel.

    Parameters
    ----------
        img : torch.Tensor
            Input image to be normalized. Shape (C, H, W)|(B, C, H, W).

    Raises
    ------
        ValueError if input image has illegal shape.

    Returns
    -------
        torch.Tensor:
            Normalized image. Shape (C, H, W)|(B, C, H, W).
    """
    if img.ndim == 4:
        _, C, _, _ = img.shape
    elif img.ndim == 3:
        C, _, _ = img.shape
    else:
        raise ValueError("img tensor shape should be either (C, H, W)|(B, C, H, W)")

    img = img.float()
    percentile1 = torch.zeros(C, dtype=img.dtype, device=img.device)
    percentile99 = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        if img.ndim == 4:
            percentile1[channel] = percentile(img[:, channel, ...], q=1)
            percentile99[channel] = percentile(img[:, channel, ...], q=99)
        else:
            percentile1[channel] = percentile(img[channel, ...], q=1)
            percentile99[channel] = percentile(img[channel, ...], q=99)

    img.sub_(percentile1.view(-1, 1, 1)).div_(
        (percentile99 - percentile1).view(-1, 1, 1)
    )

    return img


def minmax_normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """Min-max normalize image tensor per channel.

    Parameters
    ----------
        img : torch.Tensor
            Input image tensor. shape (C, H, W)|(B, C, H, W).

    Raises
    ------
        ValueError if input image has illegal shape.

    Returns
    -------
        torch.Tensor:
            Minmax normalized image tensor. Shape (C, H, W)|(B, C, H, W).
    """
    if img.ndim == 4:
        _, C, _, _ = img.shape
    elif img.ndim == 3:
        C, _, _ = img.shape
    else:
        raise ValueError("img tensor shape should be either (C, H, W)|(B, C, H, W)")

    img = img.float()
    chl_min = torch.zeros(C, dtype=img.dtype, device=img.device)
    chl_max = torch.zeros(C, dtype=img.dtype, device=img.device)
    for channel in range(C):
        if img.ndim == 4:
            chl_min[channel] = torch.min(img[:, channel, ...])
            chl_max[channel] = torch.max(img[:, channel, ...])
        else:
            chl_min[channel] = torch.min(img[channel, ...])
            chl_max[channel] = torch.max(img[channel, ...])

    return img.sub_(chl_min.view(-1, 1, 1)).div_((chl_max - chl_min).view(-1, 1, 1))


def normalize_torch(img: torch.Tensor) -> torch.Tensor:
    """Standardize image tensor per channel.

    Parameters
    ----------
        img : torch.Tensor
            Input image tensor. shape (C, H, W).

    Raises
    ------
        ValueError if input image has illegal shape.

    Returns
    -------
        torch.Tensor:
            Standardized image tensor. Shape (C, H, W)|(B, C, H, W).
    """
    if not 3 <= img.ndim <= 4:
        raise ValueError("img tensor shape should be either (C, H, W)|(B, C, H, W)")

    img = img.float()
    chl_means = torch.mean(img, dim=(-2, -1))
    chl_stds = torch.std(img, dim=(-2, -1))

    img.sub_(chl_means.view(-1, 1, 1)).div_(chl_stds.view(-1, 1, 1))

    return img


NORM_LOOKUP = {
    "minmax": minmax_normalize_torch,
    "norm": normalize_torch,
    "percentile": percentile_normalize_torch,
    "dataset": dataset_normalize_torch,
}
