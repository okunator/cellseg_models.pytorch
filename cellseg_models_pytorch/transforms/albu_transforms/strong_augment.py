from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from albumentations.core.composition import BaseCompose
    from albumentations.core.transforms_interface import (
        BasicTransform,
        ImageOnlyTransform,
    )

    HAS_ALBU = True
except ModuleNotFoundError:
    HAS_ALBU = False

from ..functional.generic_transforms import (
    AUGMENT_SPACE,
    _apply_operation,
    _check_augment_space,
    _magnitude_kwargs,
)

TransformType = Union[BasicTransform, "BaseCompose"]
TransformsSeqType = Sequence[TransformType]

__all__ = ["AlbuStrongAugment", "StrongAugTransform"]


class StrongAugTransform(ImageOnlyTransform):
    def __init__(
        self,
        operation_name: str,
        aug_space: Tuple[Any, Any],
        rng: np.random.RandomState,
        **kwargs,
    ) -> None:
        """Create StronAugment transformation.

        This is a albumentations wrapper for the StrongAugment transformations.

        Args:
            operation_name: Name of the transformation to apply.
            aug_space:
                Tuple containing the lower and upper bounds for the transformation.
            rng: Random number generator.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `StrongAugTransform` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )
        super().__init__(p=1.0)
        self.op_name = operation_name
        self.aug_space = aug_space
        self.rng = rng

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply a transformation from the StrognAugment augmentation space.

        Args:
            image: Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns:
            Transformed image. Same shape as input. dtype: float32.
        """
        kwargs = dict(
            name=self.op_name,
            **_magnitude_kwargs(self.op_name, bounds=self.aug_space, rng=self.rng),
        )
        self.params = self.update_params(kwargs)
        transformed = _apply_operation(image, self.op_name, **kwargs)
        return transformed

    def get_transform_init_args_names(self):
        """Get the names of the transformation arguments."""
        return ("op_name", "aug_space", "rng")

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Update the transformation parameters."""
        params.update({kw: it for kw, it in kwargs.items() if kw != "image"})
        return params


class AlbuStrongAugment(BaseCompose):
    """Apply a random selection of image appearance augmentations.

    Use in ``albumentations.Compose`` before normalization. It expects an RGB
    NumPy image in the uint8 range 0–255 and leaves supplied masks unchanged.
    Call with ``image=image`` and optionally ``masks=[mask, ...]``. Returns a
    dictionary containing the image and any supplied masks.

    The policy follows StrongAugment, described in
    https://arxiv.org/abs/2206.15274.
    """

    def __init__(
        self,
        augment_space: Dict[str, tuple] = AUGMENT_SPACE,
        operations: Tuple[int] = (3, 4, 5),
        probabilites: Tuple[float] = (0.2, 0.3, 0.5),
        seed: Optional[int] = None,
        p=1.0,
    ) -> None:
        """Choose augmentation operations and how many to apply per image.

        Args:
            augment_space: Dictionary mapping operation names to magnitude bounds.
                The default supplies the package's supported appearance operations.
            operations: Candidate numbers of distinct operations per image. Each
                count must fit the number of configured operations.
            probabilites: Sampling probabilities for those counts, in the same order.
                This spelling is the current public parameter name.
            seed: Seed for selecting operations and magnitudes. Operation count and
                application probability also use NumPy's global random state.
            p: Probability of applying the sampled operations.

        Raises:
            ValueError: Operation names/bounds are invalid or the operations and
                probabilites lengths differ. NumPy also validates sampling arguments.
        """
        if not HAS_ALBU:
            raise ModuleNotFoundError(
                "To use the `StrongAugment` class, the albumentations lib is needed. "
                "Install with `pip install albumentations`"
            )

        _check_augment_space(augment_space)
        if len(operations) != len(probabilites):
            raise ValueError("Operation length does not match probabilities length.")

        self.rng = np.random.RandomState(seed=seed)
        transforms = [
            StrongAugTransform(op, aug_space, self.rng)
            for op, aug_space in augment_space.items()
        ]
        self.augment_space = augment_space
        self.operations = operations
        self.probabilites = probabilites
        self.last_operations = dict()
        super().__init__(transforms, p=p)

    def __call__(self, *args, force_apply: bool = False, **data) -> Dict[str, Any]:
        """Augment one image while retaining its masks.

        Args:
            *args: Use keyword inputs; positional values are not used.
            force_apply: Apply sampled operations regardless of p.
            **data: ``image`` as an HWC uint8 RGB array, optionally ``masks`` as a list
                of aligned HW masks. Other keys are not retained.

        Returns:
            Dictionary containing ``image`` and any supplied ``masks``. Image shape
            and dtype are preserved; masks keep their labels and geometry.
        """
        image = data["image"].copy()

        masks = None
        if "masks" in data:
            masks = data["masks"].copy()

        num_ops = np.random.choice(self.operations, p=self.probabilites)
        idx = self.rng.choice(len(self.transforms), size=num_ops, replace=False)

        rs = np.random.random()
        if force_apply or rs < self.p:
            for i in idx:
                t = self.transforms[i]
                name = t.op_name
                data = t(image=image, masks=masks, force_apply=True)
                image = data["image"]
                self.last_operations[name] = t.params

        return {k: d for k, d in data.items() if k in ("image", "masks")}

    def __repr__(self) -> str:
        """Return the string representation of the StrongAugment object."""
        return (
            f"{self.__class__.__name__}("
            f"operations={self.operations}, "
            f"probabilites={self.probabilites}, "
            f"augment_space={self.augment_space})"
        )
