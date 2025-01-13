from typing import Callable, Dict, List, Tuple

try:
    import albumentations as A
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The albumentations lib is needed. Install with `pip install albumentations`"
    )

import numpy as np
import torch

__all__ = [
    "OnlyInstMapTransform",
    "ApplyEach",
    "ToTensorV3",
]


class OnlyInstMapTransform(A.BasicTransform):
    """Transforms applied to only instance labelled masks."""

    def __init__(self) -> None:
        super().__init__(p=1.0)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,  # needs to have this key or braks.
            "inst_map": self.apply_to_instmap,
        }

    def apply(self, img: np.ndarray, **params):
        return

    def apply_to_instmap(self, inst_map: np.ndarray, **params):
        raise NotImplementedError("`apply_to_instmap` method not implemented")


class ApplyEach(A.BaseCompose):
    def __init__(self, transforms: List[A.BasicTransform], p: float = 1.0) -> None:
        """Apply each transform to the input non-sequentially.

        Returns outputs for each transform.
        """
        super().__init__(transforms, p)

    def __call__(self, **data):
        res = {}
        for t in self.transforms:
            res[t.name] = t(force_apply=True, **data)

        return res


class ToTensorV3(A.BasicTransform):
    """Convert image, masks and auxilliary inputs to tensor."""

    def __init__(self):
        super().__init__(p=1.0)

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "masks": self.apply_to_masks,
            "aux": self.apply_to_aux,
        }

    def apply(self, img: np.ndarray, **params):
        if len(img.shape) not in [2, 3]:
            raise ValueError("Images need to be in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def apply_to_masks(self, masks: List[np.ndarray], **params):
        for i, mask in enumerate(masks):
            if mask.ndim == 3:
                mask = mask.transpose(2, 0, 1)
            masks[i] = torch.from_numpy(mask).long()

        return masks

    def apply_to_aux(self, auxilliary: Dict[str, np.ndarray], **params):
        for k, aux in auxilliary.items():
            if k != "binary":
                auxilliary[k] = torch.from_numpy(aux["inst_map"]).float()
            else:
                auxilliary[k] = torch.from_numpy(aux["inst_map"]).long()

        return auxilliary

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("p",)
