from typing import Callable, Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch import ToTensorV2

__all__ = ["apply_each", "compose", "to_tensor", "to_tensorv3"]


class OnlyInstMapTransform(A.BasicTransform):
    """Transforms applied to only instance labelled masks."""

    def __init__(self) -> None:
        super().__init__(always_apply=True, p=1.0)

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
        super().__init__(always_apply=True, p=1.0)

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
        return ("always_apply", "p")


def apply_each(transforms: List[OnlyInstMapTransform]) -> Callable:
    """Apply each transform wrapper.

    Example
    -------
        >>> im = read_image("/path/to/image.png")
        >>> inst_map = read_mask("/path/to/mask.mat")
        >>> tr = apply_each([cellpose_transform(), edgeweight_transform()])
        >>> aug = tr(image=im, inst_map=inst_map)
        >>> print(aug["cellpose"]["inst_map"].shape)
        (2, 256, 256)
        >>> print(aug["edgeweight"]["inst_map"].shape)
        (256, 256)

    Returns
    -------
        ApplyEach: ApplyEach object.
    """
    result = ApplyEach([item for sublist in transforms for item in sublist])

    return result


def compose(transforms_to_compose: List[A.BasicTransform]) -> Callable:
    """Compose transforms with albumentations Compose.

    Takes in a list of albumentation transforms and composes them to one
    transformation pipeline.

    Example
    -------
        >>> im = read_image("/path/to/image.png")
        >>> inst_map = read_mask("/path/to/mask.mat")
        >>> tr = compose([rigid_transform(), blur_transform(), minmax_normalize()])
        >>> aug = tr(image=im, masks=[inst_map])
        >>> print(aug["image"].shape)
        (256, 256, 3)
        >>> print(aug["masks"][0].shape)
        (256, 256)

    Returns
    -------
        Callable:
            A composed pipeline of albumentation transforms.
    """
    result = A.Compose([item for sublist in transforms_to_compose for item in sublist])
    return result


def to_tensor(**kwargs) -> List[BasicTransform]:
    """Convert each patch (np.ndarray) is into torch.Tensor.

    Returns
    -------
        List[BasicTransform]:
            A tensor conversion transform.
    """
    return [ToTensorV2()]


def to_tensorv3(**kwargs) -> BasicTransform:
    """Convert images, label-, auxilliary-, and semantic masks to tensors.

    Returns
    -------
        BasicTransform:
            A tensor conversion transform
    """
    return ToTensorV3()
