from typing import List

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from ...utils import minmax_normalize, normalize, percentile_normalize

__all__ = ["imgnorm_transform", "percentilenorm_transform", "minmaxnorm_transform"]


class MinMaxNormalization(ImageOnlyTransform):
    def __init__(
        self,
        amin: float = None,
        amax: float = None,
        always_apply: bool = True,
        p: float = 1.0,
        **kwargs
    ) -> None:
        """Min-max normalization transformation.

        Parameters
        ----------
            amin : float, optional
                Clamp min value. No clamping performed if None.
            amax : float, optional
                Clamp max value. No clamping performed if None.
        """
        super().__init__(always_apply, p)
        self.amin = amin
        self.amax = amax

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply min-max normalization.

        Parameters
        ----------
            image : np.ndarray:
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns
        -------
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return minmax_normalize(image, self.amin, self.amax)


class PercentileNormalization(ImageOnlyTransform):
    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 99.99,
        always_apply: bool = True,
        p: float = 1.0,
        **kwargs
    ) -> None:
        """Percentile normalization transformation.

        Parameters
        ----------
            amin : float, optional
                Clamp min value. No clamping performed if None.
            amax : float, optional
                Clamp max value. No clamping performed if None.
        """
        super().__init__(always_apply, p)
        self.lower = lower
        self.upper = upper

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply percentile normalization to input image.

        Parameters
        ----------
            image : np.ndarray:
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns
        -------
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return percentile_normalize(image, self.lower, self.upper)


class ImgNormalization(ImageOnlyTransform):
    def __init__(
        self,
        standardize: bool = True,
        amin: float = None,
        amax: float = None,
        always_apply: bool = True,
        p: float = 1.0,
        **kwargs
    ) -> None:
        """Image level normalization transformation.

        NOTE: this is not dataset-level normalization but image-level.

        Parameters
        ----------
            standardize : bool, default=True
                If True, divides the mean shifted img by the standard deviation.
            amin : float, optional
                Clamp min value. No clamping performed if None.
            amax : float, optional
                Clamp max value. No clamping performed if None.
        """
        super().__init__(always_apply, p)
        self.standardize = standardize
        self.amin = amin
        self.amax = amax

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply image-level normalization to input image.

        Parameters
        ----------
            image : np.ndarray:
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns
        -------
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return normalize(image, self.standardize, self.amin, self.amax)


def imgnorm_transform(**kwargs) -> List[ImageOnlyTransform]:
    """Return image-level normalization transform."""
    return [ImgNormalization(**kwargs)]


def percentilenorm_transform(**kwargs) -> List[ImageOnlyTransform]:
    """Return percentile normalization transform."""
    return [PercentileNormalization(**kwargs)]


def minmaxnorm_transform(**kwargs) -> List[ImageOnlyTransform]:
    """Return minmax normalization transform."""
    return [MinMaxNormalization(**kwargs)]
