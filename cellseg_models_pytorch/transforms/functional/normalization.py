import cv2
import numpy as np
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

    Parameters:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W).
        lower (float, default=0.01):
            The lower percentile
        upper (float, default=99.99):
            The upper percentile
        copy (bool, default=False):
            If True, normalize the copy of the input.

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
    img: np.ndarray, amin: float = None, amax: float = None, copy: bool = False
) -> np.ndarray:
    """Channelwise 1-99 percentile normalization. Optional clamping.

    Parameters:
        img (np.ndarray)
            Input image to be normalized. Shape (H, W, C)|(H, W).
        copy (bool, default=False):
            If True, normalize the copy of the input.

    Returns:
        np.ndarray:
            Normalized image. Same shape as input. dtype: float32.

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

    percentile1 = np.percentile(im, q=1, axis=axis)
    percentile99 = np.percentile(im, q=99, axis=axis)
    num = im - percentile1
    denom = percentile99 - percentile1
    im = np.divide(num, denom, where=denom != 0)

    return im.astype(np.float32)


def normalize(
    img: np.ndarray,
    mean: np.ndarray = None,
    denom: np.ndarray = None,
    copy: bool = False,
) -> np.ndarray:
    """Channelwise mean centering or standardizing of an image. Optional clamping.

    Parameters:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W).
        mean (np.ndarray, default=None):
            Channel-wise mean values to subtract from the image. Shape (C,). If None,
            the channel-wise mean of the input image is used.
        denom (np.ndarray, default=None):
            Value to divide the image by. In practice, we do a multiplication because
            it's faster. So set this to the reciprocal of intended denominator. E.g.
            the inputs, if you want to standardize, use the reciprocal of the standard
            deviation of the inputs. Shape (C,). If None, the channel-wise reciprocal
            standard deviation of the input image is used.
        copy (bool, default=False):
            If True, normalize the copy of the input.

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
    img: np.ndarray, amin: float = 0.0, amax: float = 1.0, copy: bool = False
) -> np.ndarray:
    """Min-max normalization per image channel. Optional clamping.

    Parameters:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W).
        amin (float, default=0.0):
            Clamp min value. No clamping performed if None.
        amax (float, default=1.0):
            Clamp max value. No clamping performed if None.
        copy (bool, default=False):
            If True, normalize the copy of the input.

    Returns:
        np.ndarray:
            Min-max normalized image. Same shape as input. dtype: float32.

    Raises:
        ValueError;
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

    return cv2.normalize(
        im.astype(np.float32), None, alpha=amin, beta=amax, norm_type=cv2.NORM_MINMAX
    )


def float2ubyte(mat: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Convert float64 to uint8.

    Float matrix values need to be in range [-1, 1] for img_as_ubyte so
    the image is normalized or clamped before conversion.

    Parameters:
        mat (np.ndarray):
            A float64 matrix. Shape (H, W, C).
        normalize (bool, default=False):
            Normalizes input to [0, 1] first. If not True,
            clips values between [-1, 1].

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
