# augmentations from the StrongAugment paper: https://github.com/jopo666/StrongAugment

import functools
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.random import RandomState
from PIL import Image, ImageOps

__all__ = [
    "AUGMENT_SPACE",
    "NAME_TO_OPERATION",
    "ALLOWED_OPERATIONS",
    "adjust_channel",
    "adjust_hue",
    "adjust_saturation",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_gamma",
    "solarize",
    "posterize",
    "sharpen",
    "autocontrast",
    "equalize",
    "grayscale",
    "gaussian_blur",
    "emboss",
    "jpeg",
    "add_noise",
    "tone_shift",
    "_apply_operation",
    "_check_augment_space",
    "_check_operation_bounds",
]


MAGNITUDE = Union[float, int, bool]
AUGMENT_SPACE = dict(
    red=(0.0, 2.0),
    green=(0.0, 2.0),
    blue=(0.0, 2.0),
    hue=(-0.5, 0.5),
    saturation=(0.0, 2.0),
    brightness=(0.1, 2.0),
    contrast=(0.1, 2.0),
    gamma=(0.1, 2.0),
    solarize=(0, 255),
    posterize=(1, 8),
    sharpen=(0.0, 1.0),
    emboss=(0.0, 1.0),
    blur=(0.0, 3.0),
    noise=(0.0, 0.2),
    jpeg=(0, 100),
    tone=(0.0, 1.0),
    autocontrast=(True, True),
    equalize=(True, True),
    grayscale=(True, True),
)


def adjust_channel(
    image: np.ndarray, magnitude: float, channel: int, **kwargs
) -> np.ndarray:
    """Adjust the specified channel of the image by the given magnitude."""
    image[..., channel] = cv2.addWeighted(
        image[..., channel],
        magnitude,
        np.zeros_like(image[..., channel]),
        1 - magnitude,
        gamma=0,
    )
    return image


def adjust_hue(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Adjust the hue of the image by the given magnitude."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * magnitude, 180).astype(np.uint8)
    hsv[..., 0] = cv2.LUT(hsv[..., 0], lut)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def adjust_saturation(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Adjust the saturation of the image by the given magnitude."""
    gray = grayscale(image)
    if magnitude == 0:
        return gray
    return cv2.addWeighted(image, magnitude, gray, 1 - magnitude, gamma=0)


def adjust_brightness(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Adjust the brightness of the image by the given magnitude."""
    return cv2.addWeighted(
        image, magnitude, np.zeros_like(image), 1 - magnitude, gamma=0
    )


def adjust_contrast(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Adjust the contrast of the image by the given magnitude."""
    mean = np.full_like(
        image,
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).mean(),
        dtype=image.dtype,
    )
    return cv2.addWeighted(image, magnitude, mean, 1 - magnitude, gamma=0)


def adjust_gamma(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Adjust the gamma of the image by the given magnitude."""
    table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** magnitude) * 255
    return cv2.LUT(image, table.astype(np.uint8))


def solarize(image: np.ndarray, magnitude: int, **kwargs) -> np.ndarray:
    """Solarize the image by the given magnitude."""
    lut = [(i if i < int(round(magnitude)) else 255 - i) for i in range(256)]
    return cv2.LUT(image, np.array(lut, dtype=np.uint8))


def posterize(image: np.ndarray, magnitude: int, **kwargs) -> np.ndarray:
    """Posterize the image by the given magnitude."""
    return (image & -int(2 ** (8 - int(round(magnitude))))).astype(np.uint8)


def autocontrast(image: np.ndarray, **kwargs) -> np.ndarray:
    """Autocontrast the image."""
    # histogram function is ffffast as fuck in PIL.
    return np.array(ImageOps.autocontrast(Image.fromarray(image)))


def equalize(image: np.ndarray, **kwargs) -> np.ndarray:
    """Equalize the image."""
    output = np.empty_like(image)
    for c in range(image.shape[-1]):
        output[..., c] = cv2.equalizeHist(image[..., c])
    return output


def grayscale(image: np.ndarray, **kwargs) -> np.ndarray:
    """Convert rgb image to grayscale."""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)


def gaussian_blur(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Apply gaussian blur to the image."""
    if magnitude <= 0:
        return image
    # Define kernel size.
    kernel_size = round(float(magnitude) * 3.5)
    kernel_size = max(3, kernel_size // 2 * 2 + 1)
    return cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=magnitude)


def sharpen(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Sharpen image."""
    kernel_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel_sharpen = np.array(
        [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
        dtype=np.float32,
    )
    kernel = (1 - magnitude) * kernel_nochange + magnitude * kernel_sharpen
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def emboss(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Emboss image."""
    kernel_nochange = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    kernel = (1 - magnitude) * kernel_nochange + magnitude * kernel_emboss
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)


def jpeg(image: np.ndarray, magnitude: int, **kwargs) -> np.ndarray:
    """Apply jpeg compression to the image."""
    return cv2.imdecode(
        cv2.imencode(".jpeg", image, (cv2.IMWRITE_JPEG_QUALITY, int(round(magnitude))))[
            1
        ],
        cv2.IMREAD_UNCHANGED,
    )


def tone_shift(image, magnitude_0: float, magnitude_1: float, **kwargs) -> np.ndarray:
    """Apply tone shift to the image."""
    t = np.linspace(0.0, 1.0, 256)
    evaluate_bez = np.vectorize(
        lambda t: 3 * (1 - t) ** 2 * t * magnitude_0
        + 3 * (1 - t) * t**2 * magnitude_1
        + t**3
    )
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)
    return cv2.LUT(image, lut=remapping)


def add_noise(image: np.ndarray, magnitude: float, **kwargs) -> np.ndarray:
    """Add noise to the image."""
    noise = np.random.randint(0, 255, size=image.shape[:2], dtype=np.uint8)
    for c in range(3):
        image[..., c] = cv2.addWeighted(
            image[..., c], 1 - magnitude, noise, magnitude, gamma=0.0
        )
    return image


NAME_TO_OPERATION = dict(
    red=functools.partial(adjust_channel, channel=0),
    green=functools.partial(adjust_channel, channel=1),
    blue=functools.partial(adjust_channel, channel=2),
    hue=adjust_hue,
    saturation=adjust_saturation,
    brightness=adjust_brightness,
    contrast=adjust_contrast,
    gamma=adjust_gamma,
    solarize=solarize,
    posterize=posterize,
    sharpen=sharpen,
    emboss=emboss,
    blur=gaussian_blur,
    noise=add_noise,
    jpeg=jpeg,
    tone=tone_shift,
    autocontrast=autocontrast,
    equalize=equalize,
    grayscale=grayscale,
)
ALLOWED_OPERATIONS = list(NAME_TO_OPERATION.keys())


def _apply_operation(image: np.ndarray, operation_name: str, **kwargs) -> np.ndarray:
    operation_fn = NAME_TO_OPERATION.get(operation_name.lower())
    if operation_fn is None:
        raise ValueError(
            f"Operation '{operation_name.lower()}' not supported. Please, "
            f"select from:\n{ALLOWED_OPERATIONS}."
        )
    return operation_fn(image, **kwargs)


def _check_augment_space(space: Dict[str, Tuple[MAGNITUDE, MAGNITUDE]]) -> None:
    """Check that passed augmentation space is valid."""
    if not isinstance(space, dict):
        raise TypeError(f"Augment space should be a dict, not {type(space)}")
    for key, val in space.items():
        if key not in ALLOWED_OPERATIONS:
            raise ValueError(
                f"Operation '{key}' not supported. Select from: {ALLOWED_OPERATIONS}"
            )
        if not isinstance(val, tuple) or len(val) != 2:
            raise TypeError("Bounds should be a (low, high) tuple.")
        # Check bounds.
        low, high = val
        if type(low) is not type(high):
            raise TypeError(
                f"Bound types should be the same ({type(low)} != {type(high)})"
            )
        elif not isinstance(low, (int, float, bool)):
            raise TypeError(f"Bounds should be int/float/bool, not {type(low)}")
        _check_operation_bounds(key, low, high)


def _check_operation_bounds(name: str, low: MAGNITUDE, high: MAGNITUDE) -> None:
    # Check operation types.
    TYPE_MSG = "Bounds for operation '{}' should be {}, not {}."
    if not isinstance(low, int) and name in ["solarize", "posterize", "jpeg"]:
        raise ValueError(TYPE_MSG.format(name, int, type(low)))
    elif not isinstance(low, bool) and name in [
        "autocontrast",
        "equalize",
        "grayscale",
    ]:
        raise ValueError(TYPE_MSG.format(name, bool, type(low)))
    # Check operation bounds.
    BOUND_MSG = "Bounds for operation '{}' should be between [{}]."
    if name != "hue" and low < 0:
        raise ValueError(f"Negative values are not allowed for operation '{name}'")
    elif name == "hue" and (low < -0.5 or high > 0.5):
        raise ValueError(BOUND_MSG.format(name, (-0.5, 0.5)))
    elif name == "solarize" and high > 256:
        raise ValueError(BOUND_MSG.format(name, (0, 256)))
    elif name == "posterize" and high > 8:
        raise ValueError(BOUND_MSG.format(name, (0, 8)))
    elif name == "jpeg" and high > 100:
        raise ValueError(BOUND_MSG.format(name, (0, 100)))
    elif name == "tone" and high > 1.0:
        raise ValueError(BOUND_MSG.format(name, (0, 1.0)))


def _magnitude_kwargs(
    operation_name: str, bounds: Tuple[MAGNITUDE, MAGNITUDE], rng: RandomState
) -> Optional[Dict[str, MAGNITUDE]]:
    """Generate magnitude kwargs for apply_operations."""
    if operation_name == "tone":
        return dict(
            magnitude_0=_sample_magnitude(*bounds, rng),
            magnitude_1=_sample_magnitude(*bounds, rng),
        )
    magnitude = _sample_magnitude(*bounds, rng)
    if magnitude is None:
        return dict()
    else:
        return dict(magnitude=magnitude)


def _sample_magnitude(low: MAGNITUDE, high: MAGNITUDE, rng: RandomState) -> MAGNITUDE:
    """Sample magnitude value."""
    if isinstance(low, float):
        return rng.uniform(low, high)
    elif isinstance(low, int):
        return rng.choice(range(low, high + 1))
    else:
        # Boolean does not require arguments.
        return None
