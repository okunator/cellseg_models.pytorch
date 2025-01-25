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

        Parameters
        ----------
            operation_name : str
                Name of the transformation to apply.
            aug_space : Tuple[Any, Any]
                Tuple containing the lower and upper bounds for the transformation.
            rng : np.random.RandomState
                Random number generator.
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

        Parameters
        ----------
            image : np.ndarray:
                Input image to be normalized. Shape (H, W, C)|(H, W).

        Returns
        -------
            np.ndarray:
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
    def __init__(
        self,
        augment_space: Dict[str, tuple] = AUGMENT_SPACE,
        operations: Tuple[int] = (3, 4, 5),
        probabilites: Tuple[float] = (0.2, 0.3, 0.5),
        seed: Optional[int] = None,
        p=1.0,
    ) -> None:
        """Strong augment augmentation policy albumentations wrapper.

        Augment like there's no tomorrow: Consistently performing neural networks for
        medical imaging: https://arxiv.org/abs/2206.15274

        Parameters
        ----------
            augment_space : Dict[str, tuple], default: AUGMENT_SPACE
                Augmentation space to sample operations from.
            operations : Tuple[int], default: [3, 4, 5].
                Number of operations to apply. If None, sample from
                [1, len(augment_space)].
            probabilites : Tuple[float], default: [0.2, 0.3, 0.5]
                Probabilities of sampling operations. If None, sample from
                the uniform distribution.
            seed : Optional[int], default: None
                Random seed.
            p : float, default: 1.0
                Probability of applying the transform.
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
        """Apply the StrongAugment transformation pipeline."""
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
