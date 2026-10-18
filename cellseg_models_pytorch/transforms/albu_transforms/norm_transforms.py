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
    "Normalization",
]


class MinMaxNormalization(ImageOnlyTransform):
    """Rescale image values using one minimum and maximum across all channels.

    Use in ``albumentations.Compose`` and call with ``image=image``. The returned
    dictionary contains a float32 image of the same shape when applied; masks
    are passed through unchanged. Input images use height, width, channels axes.
    """

    def __init__(
        self,
        amin: float = 0.0,
        amax: float = 1.0,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Choose the output range for global min-max normalization.

        Args:
            amin: One end of the output range, normally 0.
            amax: Other end, normally 1. Reversed endpoints are accepted. Constant
                images use the lower endpoint.
            p: Probability of applying this image-only transform.
            copy: Copy the NumPy input before normalization. The input is not modified.
            **kwargs: Additional arguments are accepted but unused.

        Example:
            >>> import albumentations as A
            >>> import numpy as np
            >>> from cellseg_models_pytorch.transforms.albu_transforms import (
            ...     MinMaxNormalization,
            ... )
            >>> transform = A.Compose([MinMaxNormalization(copy=True)])
            >>> result = transform(image=np.array([[0, 255]], dtype=np.uint8))
            >>> result["image"].tolist()
            [[0.0, 1.0]]
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

        Args:
            image: Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return minmax_normalize(image, self.amin, self.amax, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("amin", "amax")


class PercentileNormalization(ImageOnlyTransform):
    """Rescale image values using global lower and upper percentiles.

    Use in ``albumentations.Compose`` and call with ``image=image``. The returned
    dictionary contains a float32 image of the same shape when applied; masks
    are passed through unchanged. Input images use height, width, channels axes.
    """

    def __init__(
        self,
        lower: float = 0.01,
        upper: float = 99.99,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Choose the percentiles mapped to 0 and 1.

        Args:
            lower: Lower percentile across all pixels and channels, from 0 to 100.
            upper: Upper percentile across all pixels and channels, from 0 to 100.
                Values beyond the chosen interval are clipped by interpolation.
            p: Probability of applying the transform.
            copy: Accepted by the underlying function, which currently copies input
                regardless of this value.
            **kwargs: Additional arguments are accepted but unused.

        Note:
            This transform uses global percentiles, not separate channel percentiles.
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

        Args:
            image: Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return percentile_normalize(image, self.lower, self.upper, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("lower", "upper")


class Normalization(ImageOnlyTransform):
    """Normalize images using supplied channel means and scale values.

    Use in ``albumentations.Compose`` and call with ``image=image``. The returned
    dictionary contains a float32 image of the same shape when applied; masks
    are passed through unchanged. Input images use height, width, channels axes.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        p: float = 1.0,
        copy: bool = False,
        **kwargs,
    ) -> None:
        """Choose the channel subtraction and scaling values.

        Args:
            mean: Per-channel values in input pixel units to subtract, shape (C,).
            std: Per-channel values, shape (C,). Applied as
                ``(image - mean) / (255 * std)``; mean is not multiplied by 255.
                Match these values and input range to the checkpoint preprocessing.
            p: Probability of applying the transform.
            copy: Copy input before calling the normalization helper. Output is a
                new float32 array.
            **kwargs: Additional arguments are accepted but unused.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `Normalization` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )
        super().__init__(p)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.reciprocal(np.array(std, dtype=np.float32) * 255.0)
        self.copy = copy

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply image-level normalization to input image.

        Args:
            image: Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            np.ndarray:
                Normalized image. Same shape as input. dtype: float32.
        """
        return normalize(image, self.mean, self.std, self.copy)

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("mean", "std")
