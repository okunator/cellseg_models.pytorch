import cv2
import numpy as np
import torch
from skimage import img_as_ubyte

__all__ = [
    "percentile_normalize",
    "percentile_normalize99",
    "normalize",
    "minmax_normalize",
    "float2ubyte",
]


def percentile_normalize(
    img: np.ndarray, lower: float = 0.01, upper: float = 99.99, copy: bool = False
) -> np.ndarray:
    """Channelwise percentile normalization to range [0, 1].

    Args:
        img: Input image to be normalized. Shape (H, W, C)|(H, W).
        lower: The lower percentile
            Defaults to 0.01.
        upper: The upper percentile
            Defaults to 99.99.
        copy: If True, normalize the copy of the input.
            Defaults to False.

    Returns:
        np.ndarray:
            Normalized img. Same shape as input. dtype: float32.

    Raises:
        ValueError:
            If input image does not have shape (H, W) or (H, W, C).
    """
    axis = (0, 1)

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    if copy:
        im = img.copy()
    else:
        im = img

    im = img.copy()

    upercentile = np.percentile(im, upper)
    lpercentile = np.percentile(im, lower)

    return np.interp(im, (lpercentile, upercentile), axis).astype(np.float32)


def percentile_normalize99(
    img: np.ndarray,
    amin: float | None = None,
    amax: float | None = None,
    copy: bool = False,
) -> np.ndarray:
    """Normalize each channel using its 1st and 99th spatial percentiles.

    A zero percentile span maps to zero before optional clipping. Without
    clipping, values outside the percentile interval can fall outside [0, 1].

    Args:
        img: Input image with HW or HWC axes, reduced over H and W.
        amin: Optional lower bound for the normalized output.
        amax: Optional upper bound for the normalized output.
        copy: If True, normalize the copy of the input.
            Defaults to False.

    Returns:
        Normalized float32 image with the original shape. Constant channels
        are zero unless clipping moves zero to a requested bound.

    Raises:
        ValueError: If the image does not have HW or HWC axes, or amin > amax.

    Examples:
        >>> percentile_normalize99(np.array([[0, 1], [2, 3]]), 0, 1).round(2)
        array([[0.  , 0.33],
               [0.67, 1.  ]], dtype=float32)
    """
    axis = (0, 1)

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    if amin is not None and amax is not None and amin > amax:
        raise ValueError("amin must be less than or equal to amax")

    if copy:
        im = img.copy()
    else:
        im = img

    percentile1 = np.percentile(im, q=1, axis=axis)
    percentile99 = np.percentile(im, q=99, axis=axis)
    num = im - percentile1
    denom = percentile99 - percentile1
    im = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    if amin is not None or amax is not None:
        np.clip(im, amin, amax, out=im)

    return im.astype(np.float32)


def normalize(
    img: np.ndarray,
    mean: np.ndarray = None,
    denom: np.ndarray = None,
    copy: bool = False,
) -> np.ndarray:
    """Channelwise mean centering or standardizing of an image. Optional clamping.

    Args:
        img: Input image to be normalized. Shape (H, W, C)|(H, W).
        mean:
            Channel-wise mean values to subtract from the image. Shape (C,). If None,
            the channel-wise mean of the input image is used.
            Defaults to None.
        denom:
            Value to divide the image by. In practice, we do a multiplication because
            it's faster. So set this to the reciprocal of intended denominator. E.g.
            the inputs, if you want to standardize, use the reciprocal of the standard
            deviation of the inputs. Shape (C,). If None, the channel-wise reciprocal
            standard deviation of the input image is used.
            Defaults to None.
        copy: If True, normalize the copy of the input.
            Defaults to False.

    Returns:
        np.ndarray:
            Normalized image. Same shape as input. dtype: float32.

    Raises:
        ValueError:
            If input image does not have shape (H, W) or (H, W, C).
    """
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    if copy:
        im = img.copy()
    else:
        im = img

    im = im.astype(np.float32)
    mean_img = np.zeros_like(im, dtype=np.float32)
    denom_img = np.zeros_like(im, dtype=np.float32)

    if mean is None:
        mean = im.mean(axis=(0, 1))
    if denom is None:
        denom = 1 / im.std(axis=(0, 1))

    mean_img = (mean_img + mean).astype(np.float32)
    denom_img = denom_img + denom

    im = cv2.subtract(im, mean_img)
    return cv2.multiply(im, denom_img, dtype=cv2.CV_32F)


def minmax_normalize(
    img: np.ndarray | torch.Tensor,
    amin: float = 0.0,
    amax: float = 1.0,
    copy: bool = False,
) -> np.ndarray | torch.Tensor:
    """Scale the global image minimum and maximum to the requested range.

    Args:
        img: HWC or HW image. Tensors stay on their input device.
        amin: Lower output value. Constant images use this value.
            Defaults to 0.0.
        amax: Upper output value.
            Defaults to 1.0.
        copy: Copy the NumPy input before normalization. Tensor results are
            always newly allocated. Defaults to False.

    Returns:
        Min-max normalized image of the same shape and array/tensor type,
        with float32 dtype. The input is not modified.

    Raises:
        ValueError: Input image does not have shape (H, W) or (H, W, C).
    """
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    if isinstance(img, torch.Tensor):
        im = img.to(dtype=torch.float32)
        minimum = im.amin()
        span = im.amax() - minimum
        low, high = min(amin, amax), max(amin, amax)
        # Match OpenCV's global range and constant-image behavior on the device.
        epsilon = torch.finfo(torch.float64).eps
        scale = torch.where(span > epsilon, (high - low) / span.clamp_min(epsilon), 0)
        return (im - minimum) * scale + low

    if copy:
        im = img.copy()
    else:
        im = img

    return cv2.normalize(
        im.astype(np.float32), None, alpha=amin, beta=amax, norm_type=cv2.NORM_MINMAX
    )


def float2ubyte(mat: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Convert float64 to uint8.

    Float matrix values need to be in range [-1, 1] for img_as_ubyte so
    the image is normalized or clamped before conversion.

    Args:
        mat: A float64 matrix. Shape (H, W, C).
        normalize: Normalizes input to [0, 1] first. If not True,
            clips values between [-1, 1].
            Defaults to False.

    Returns:
        np.ndarray:
            A uint8 matrix. Shape (H, W, C). dtype: uint8.
    """
    m = mat.copy()

    if normalize:
        m = minmax_normalize(m)
    else:
        m = np.clip(m, a_min=-1, a_max=1)

    return img_as_ubyte(m)
