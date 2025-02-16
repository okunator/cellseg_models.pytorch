import numpy as np

try:
    from albumentations.core.transforms_interface import ImageOnlyTransform

    HAS_ALBU = True
except ModuleNotFoundError:
    HAS_ALBU = False

from ..functional.normalization import minmax_normalize, normalize, percentile_normalize

__all__ = [
    "MinMaxNormalization",
    "PercentileNormalization",
    "ImgNormalization",
]


class MinMaxNormalization(ImageOnlyTransform):
    def __init__(
        self,
        amin: float = None,
        amax: float = None,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Min-max normalization transformation.

        Parameters:
            amin (float, default=None)
                Clamp min value. No clamping performed if None.
            amax (float, default=None)
                Clamp max value. No clamping performed if None.
            p (float, default=1.0):
                Probability of applying the transformation.
            copy (bool, default=False):
                If True, normalize the copy of the input.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `MinMaxNormalization` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )

        super().__init__(p)
        self.amin = amin
        self.amax = amax
        self.copy = copy

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply min-max normalization.

        Parameters:
            image (np.ndarray):
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return minmax_normalize(image, self.amin, self.amax, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("amin", "amax")


class PercentileNormalization(ImageOnlyTransform):
    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 99.99,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Percentile normalization transformation.

        Parameters:
            amin (float, default=None):
                Clamp min value. No clamping performed if None.
            amax (float, default=None):
                Clamp max value. No clamping performed if None.
            p (float, default=1.0):
                Probability of applying the transformation.
            copy (bool, default=False):
                If True, normalize the copy of the input.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `PercentileNormalization` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )

        super().__init__(p)
        self.lower = lower
        self.upper = upper
        self.copy = copy

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply percentile normalization to input image.

        Parameters:
            image (np.ndarray):
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return percentile_normalize(image, self.lower, self.upper, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("lower", "upper")


class ImgNormalization(ImageOnlyTransform):
    def __init__(
        self,
        standardize: bool = True,
        amin: float = None,
        amax: float = None,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Image level normalization transformation.

        NOTE: this is not dataset-level normalization but image-level.

        Parameters:
            standardize (bool, default=True):
                If True, divides the mean shifted img by the standard deviation.
            amin (float, default=None):
                Clamp min value. No clamping performed if None.
            amax (float, default=None):
                Clamp max value. No clamping performed if None.
            always_apply (bool, default=True):
                Apply the transformation always.
            p (float, default=1.0):
                Probability of applying the transformation.
            copy (bool, default=False):
                If True, normalize the copy of the input.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `ImgNormalization` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )
        super().__init__(p)
        self.standardize = standardize
        self.amin = amin
        self.amax = amax
        self.copy = copy

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply image-level normalization to input image.

        Parameters:
            image (np.ndarray):
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return normalize(image, self.standardize, self.amin, self.amax, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("amin", "amax", "standardize")
