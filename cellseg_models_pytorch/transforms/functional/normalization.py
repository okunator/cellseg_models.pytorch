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
    """Rescale global image percentiles to the range 0–1.

    Use one pair of percentiles across all pixels and channels. This is different
    from percentile_normalize99, which computes separate channel percentiles.

    Args:
        img: NumPy HW or HWC image; H is height, W width and C channels.
        lower: Lower percentile, in the range 0–100, mapped to 0.
        upper: Upper percentile, in the range 0–100, mapped to 1. Values outside
            the chosen interval are clipped by interpolation.
        copy: Retained option. The current implementation copies input regardless.

    Returns:
        A new float32 NumPy image with the original shape. The input is unchanged.
        If both percentile endpoints coincide, interpolation maps that value to 1.

    Raises:
        ValueError: The image does not have HW or HWC axes; NumPy also validates
            percentile arguments.
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
        img: NumPy image with (height, width) or (height, width, channels) axes.
            Percentiles are computed over height and width for each channel.
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
    """Subtract a mean and multiply by a scale for each image channel.

    Args:
        img: NumPy HW or HWC image; H is height, W width and C channels.
        mean: Per-channel values to subtract, shape (C,). None uses the input's
            own channel means. For a grayscale image, a scalar is accepted.
        denom: Per-channel multiplier, not divisor. To standardize, supply
            1 / standard_deviation. None calculates this from the input image.
            A constant channel has zero standard deviation; supply a finite
            scale explicitly if such channels are possible.
        copy: Copy input before conversion. The returned float32 array is new
            regardless; the input is not modified.

    Returns:
        Float32 NumPy array with the original shape. No clipping is applied;
        output values need not lie between 0 and 1.

    Raises:
        ValueError: The input does not have HW or HWC axes.

    Example:
        >>> image = np.array([[1, 3]], dtype=np.float32)
        >>> normalize(image, mean=np.array(2), denom=np.array(0.5)).tolist()
        [[-0.5, 0.5]]
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
    """Rescale an image's pixel values to a chosen range, usually 0 to 1.

    Use one minimum and maximum across the whole image, including all colour
    channels. This differs from normalizing each channel separately.

    Args:
        img: Image with shape (height, width, channels) or (height, width).
            NumPy arrays and PyTorch tensors are accepted. Tensors stay on their
            input device.
        amin: One output endpoint. Constant images use the lower endpoint.
        amax: Other output endpoint; the endpoints may be supplied in either order.
        copy: Copy the NumPy input before normalization. Tensor results are
            always newly allocated. Defaults to False.

    Returns:
        Min-max normalized image of the same shape and array/tensor type,
        with float32 dtype. The input is not modified.

    Raises:
        ValueError: Input image does not have shape (H, W) or (H, W, C).

    Example:
        >>> import numpy as np
        >>> from cellseg_models_pytorch.transforms import minmax_normalize
        >>> image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        >>> normalized = minmax_normalize(image)
        >>> normalized.shape == image.shape
        True
        >>> np.allclose([normalized.min(), normalized.max()], [0, 1], atol=1e-7)
        True
    """
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    if isinstance(img, torch.Tensor):
        im = img.to(dtype=torch.float32)
        minimum = im.amin()
        low, high = min(amin, amax), max(amin, amax)
        # Match OpenCV's global range and constant-image behavior on the device.
        epsilon = torch.finfo(torch.float64).eps
        if img.is_floating_point():
            # Halve before subtraction so even opposite float32 extremes fit.
            half_minimum = minimum * 0.5
            half_span = im.amax() * 0.5 - half_minimum
            unit_range = (im * 0.5 - half_minimum) / half_span.clamp_min(epsilon * 0.5)
            scale = torch.where(half_span > epsilon * 0.5, high - low, 0)
            return unit_range * scale + low

        span = im.amax() - minimum
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
    """Convert floating image values to uint8 pixel values.

    Args:
        mat: Floating NumPy image. Output keeps its shape.
        normalize: True first applies global min-max normalization to 0–1.
            False clips to -1–1 before conversion; negative values become zero.

    Returns:
        A new uint8 image with values 0–255. The input is unchanged.
    """
    m = mat.copy()

    if normalize:
        m = minmax_normalize(m)
    else:
        m = np.clip(m, a_min=-1, a_max=1)

    return img_as_ubyte(m)
