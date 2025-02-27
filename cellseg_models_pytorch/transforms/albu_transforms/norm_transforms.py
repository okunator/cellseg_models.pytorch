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
        amin: float = 0.0,
        amax: float = 1.0,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Min-max normalization. Normalizes to range [amin, amax].

        Parameters:
            amin (float, default=0.0)
                Normalization lower limit.
            amax (float, default=1.0)
                Normalization upper limit.
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
        """Percentile normalization. Normalizes to percentile range [lower, upper].

        Parameters:
            lower (float, default=0.01):
                Lower percentile.
            upper (float, default=99.99):
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


class Normalization(ImageOnlyTransform):
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Image level normalization transformation.

        NOTE: this is not dataset-level normalization but image-level.

        Parameters:
            mean (np.ndarray):
                Mean values for each channel. Shape (C,)
            std (np.ndarray):
                Standard deviation values for each channel. Shape (C,)
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
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.reciprocal(np.array(std, dtype=np.float32) * 255.0)
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
        return normalize(image, self.mean, self.std, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("mean", "std")
