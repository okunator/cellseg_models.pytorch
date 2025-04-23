from typing import List

try:
    import albumentations as A
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The albumentations lib is needed. Install with `pip install albumentations`"
    )

import numpy as np

__all__ = ["OnlyInstMapTransform", "ApplyEach"]


class OnlyInstMapTransform:
    """Transforms applied to only instance labelled masks."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def available_keys(self) -> set[str]:
        """Returns set of available keys."""
        return set(["inst", "cyto_inst"])


class ApplyEach(A.BaseCompose):
    def __init__(
        self,
        transforms: List[OnlyInstMapTransform],
        p: float = 1.0,
        as_list: bool = False,
        **kwargs,
    ) -> None:
        """Apply each transform to the input non-sequentially.

        Returns outputs for each transform.

        Parameters:
            transforms (List[Any]):
                List of transforms to apply.
            p (float, default=1.0):
                Probability of applying the transform.
            as_list (bool, default=False):
                Return the outputs as list with shapes (H, W, C).
        """
        super().__init__(transforms, p, **kwargs)
        self.names = [t.name for t in transforms]
        self.out_channels = [t.out_channels for t in transforms]
        self.as_list = as_list

    def __call__(self, **data):
        res = {}
        for t in self.transforms:
            for k, d in data.items():
                res[f"{k}-{t.name}"] = t(d, force_apply=True)

        if self.as_list:
            masks = []
            for k, v in res.items():
                if v.ndim == 2:
                    v = v[np.newaxis]
                masks.append(v.transpose(1, 2, 0))
            return masks

        return res

    def __repr__(self) -> str:
        """Return a string representation of the class."""
        return f"{self.__class__.__name__}({self.names})"
