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
        amin (float, default=None)
            Clamp min value. No clamping performed if None.
        amax (float, default=None):
            Clamp max value. No clamping performed if None.
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
    im = num / denom

    # clamp
    if not any(x is None for x in (amin, amax)):
        im = np.clip(im, a_min=amin, a_max=amax)

    return im.astype(np.float32)


def normalize(
    img: np.ndarray,
    standardize: bool = True,
    amin: float = None,
    amax: float = None,
    copy: bool = False,
) -> np.ndarray:
    """Channelwise mean centering or standardizing of an image. Optional clamping.

    Parameters:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W).
        standardize (bool, default=True):
            If True, divide with standard deviation after mean centering
        amin (float, default=None):
            Clamp min value. No clamping performed if None.
        amax (float, default=None):
            Clamp max value. No clamping performed if None.
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

    # mean center
    im = im - im.mean(axis=axis, keepdims=True)

    if standardize:
        std = im.std(axis=axis, keepdims=True)
        im = np.divide(im, std, where=std != 0)

    # clamp
    if not any(x is None for x in (amin, amax)):
        im = np.clip(im, a_min=amin, a_max=amax)

    return im.astype(np.float32)


def minmax_normalize(
    img: np.ndarray, amin: float = None, amax: float = None, copy: bool = False
) -> np.ndarray:
    """Min-max normalization per image channel. Optional clamping.

    Parameters:
        img (np.ndarray):
            Input image to be normalized. Shape (H, W, C)|(H, W).
        amin (float, default=None):
            Clamp min value. No clamping performed if None.
        amax (float, default=None):
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

    min = im.min()
    max = im.max()
    denom = max - min
    num = im - min
    im = num / denom

    # clamp
    if not any(x is None for x in (amin, amax)):
        im = np.clip(im, a_min=amin, a_max=amax)

    return im.astype(np.float32)


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
