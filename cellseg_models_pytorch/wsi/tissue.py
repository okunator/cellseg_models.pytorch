"""Ported from:
https://github.com/jopo666/HistoPrep/tree/master/histoprep/functional/_tissue.py

MIT License

Copyright (c) 2023 jopo666

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Optional

import cv2
import numpy as np

ERROR_THRESHOLD = "Threshold should be in range [0, 255], got {}."
ERROR_TYPE = "Expected an np.ndarray image, not {}."
ERROR_DIMENSIONS = "Image should have 2 or 3 dimensions, not {}."
ERROR_CHANNELS = "Image should have 3 colour channels, not {}."
ERROR_DTYPE = "Expected image dtype to be uint8, not {}."

MAX_THRESHOLD = 255
WHITE_PIXEL = 255
BLACK_PIXEL = 0
SIGMA_NO_OP = 0.0
GRAY_NDIM = 2
RGB_NDIM = 3


def check_image(image: np.ndarray) -> np.ndarray:
    """Check that input is a valid RGB/L image and convert to it to an array."""
    if not isinstance(image, np.ndarray):
        raise TypeError(ERROR_TYPE.format(type(image)))
    if not (image.ndim == GRAY_NDIM or image.ndim == RGB_NDIM):
        raise TypeError(ERROR_DIMENSIONS.format(image.ndim))
    if image.ndim == RGB_NDIM and image.shape[-1] != RGB_NDIM:
        raise TypeError(ERROR_CHANNELS.format(image.shape[-1]))
    if image.dtype != np.uint8:
        raise TypeError(ERROR_DTYPE.format(image.dtype))
    return image


def get_tissue_mask(
    image: np.ndarray,
    *,
    threshold: Optional[int] = None,
    multiplier: float = 1.0,
    sigma: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Detect tissue from image.

    Parameters:
        image (np.ndarray):
            Input image.
        threshold (int, default=None):
            Threshold for tissue detection. If set, will detect tissue by global
            thresholding, and otherwise Otsu's method is used to find a threshold.
        multiplier (float, default=1.0):
            Otsu's method is used to find an optimal threshold by minimizing the weighted
            within-class variance. This threshold is then multiplied with `multiplier`.
            Ignored if `threshold` is not None.
        sigma (float, default=1.0):
            Sigma for gaussian blurring. Defaults to 1.0.

    Raises:
        ValueError: Threshold not between 0 and 255.

    Returns:
        tuple[int, np.ndarray]:
            Tuple with `threshold` and `tissue_mask` (0=background and 1=tissue).
    """
    # Check image and convert to array.
    image = check_image(image)
    # Check arguments.
    if threshold is not None and not 0 <= threshold <= MAX_THRESHOLD:
        raise ValueError(ERROR_THRESHOLD.format(threshold))
    # Convert to grayscale.
    gray = image if image.ndim == GRAY_NDIM else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian blurring.
    blur = _gaussian_blur(image=gray, sigma=sigma, truncate=3.5)
    # Get threshold.
    if threshold is None:
        threshold = _otsu_threshold(gray=blur)
        threshold = max(min(255, int(threshold * max(0.0, multiplier) + 0.5)), 0)
    # Global thresholding.
    thrsh, mask = cv2.threshold(blur, threshold, 1, cv2.THRESH_BINARY_INV)
    return int(thrsh), mask


def clean_tissue_mask(
    tissue_mask: np.ndarray,
    min_area_pixel: int = 10,
    max_area_pixel: Optional[int] = None,
    min_area_relative: float = 0.2,
    max_area_relative: Optional[float] = 2.0,
) -> np.ndarray:
    """Remove too small/large contours from tissue mask.

    Args:
        tissue_mask: Tissue mask to be cleaned.
        min_area_pixel: Minimum pixel area for contours. Defaults to 10.
        max_area_pixel: Maximum pixel area for contours. Defaults to None.
        min_area_relative: Relative minimum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`min_area_relative * median(contour_areas)`). Defaults to 0.2.
        max_area_relative: Relative maximum contour area, calculated from the median
            contour area after filtering contours with `[min,max]_pixel` arguments
            (`max_area_relative * median(contour_areas)`). Defaults to 2.0.

    Returns:
        Tissue mask with too small/large contours removed.
    """
    contours, __ = cv2.findContours(
        tissue_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return tissue_mask
    contour_areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    # Filter based on pixel values.
    selection = contour_areas >= min_area_pixel
    if max_area_pixel is not None:
        selection = selection & (contour_areas <= max_area_pixel)
    if selection.sum() == 0:
        # Nothing to draw
        return np.zeros_like(tissue_mask)
    # Define relative min/max values.
    area_median = np.median(contour_areas[selection])
    area_min = area_median * min_area_relative
    area_max = None if max_area_relative is None else area_median * max_area_relative
    # Draw new mask.
    new_mask = np.zeros_like(tissue_mask)
    for select, area, cnt in zip(selection, contour_areas, contours):
        if select and area >= area_min and (area_max is None or area <= area_max):
            cv2.drawContours(new_mask, [cnt], -1, 1, -1)
    return new_mask


def _otsu_threshold(*, gray: np.ndarray) -> int:
    """Helper function to calculate Otsu's thresold from a grayscale image."""
    values = gray.flatten()
    values = values[(values != WHITE_PIXEL) & (values != BLACK_PIXEL)]
    threshold, __ = cv2.threshold(
        values, None, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return threshold


def _gaussian_blur(
    *, image: np.ndarray, sigma: float, truncate: float = 3.5
) -> np.ndarray:
    """Apply gaussian blurring."""
    if sigma <= SIGMA_NO_OP:
        return image
    ksize = int(truncate * sigma + 0.5)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
