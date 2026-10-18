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
    """Generate several training targets independently from the same instance mask.

    This is not a sequence of spatial augmentations. Apply joint image/mask
    augmentation first, then derive all targets from the resulting instance mask.

    Attributes:
        names: Target names in the configured transform order.
        out_channels: Channel count corresponding to each name.
    """

    def __init__(
        self,
        transforms: List[OnlyInstMapTransform],
        p: float = 1.0,
        as_list: bool = False,
        **kwargs,
    ) -> None:
        """Choose target generators and the output layout.

        Args:
            transforms: Configured instance-mask transforms, such as CellposeTransform.
            p: Retained composition probability; this implementation always calls
                each configured target generator.
            as_list: Return HWC arrays in a list, as expected by the training datasets.
                False returns a dictionary instead.
            **kwargs: Additional Albumentations BaseCompose options.
        """
        super().__init__(transforms, p, **kwargs)
        self.names = [t.name for t in transforms]
        self.out_channels = [t.out_channels for t in transforms]
        self.as_list = as_list

    def __call__(self, **data):
        """Generate every configured target for each supplied mask.

        Args:
            **data: Named integer HW instance masks, such as ``inst=mask``. Zero is
                background and positive labels identify objects.

        Returns:
            With as_list=False, a dictionary keyed as ``input-name``, for example
            ``inst-cellpose``; vector targets are CHW and scalar targets HW.
            With as_list=True, a list of HWC arrays, adding a singleton channel for
            scalar targets. Ordering is by transform, then input keyword order.
        """

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
