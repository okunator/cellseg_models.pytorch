from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.utils import FileHandler, to_tensor

if TYPE_CHECKING:
    import albumentations as A

    from cellseg_models_pytorch.transforms.albu_transforms import ApplyEach

__all__ = ["TrainDatasetH5"]


class TrainDatasetH5(Dataset):
    def __init__(
        self,
        path: str,
        img_key: str,
        inst_keys: Tuple[str, ...],
        mask_keys: Tuple[str, ...],
        transforms: A.Compose,
        inst_transforms: ApplyEach,
    ) -> None:
        """HDF5 train dataset for cell/panoptic segmentation models.

        Args:
            path: Path to the h5 file.
            img_key: Key for the image data in the h5 file.
            inst_keys:
                Key for the instance data in the h5 file. This will be transformed
            mask_keys: Keys for the semantic masks in the h5 file.
            transforms: Albumentations compose object for image and mask transforms.
            inst_transforms: ApplyEach object with as_list=True for instance transforms.

        Raises:
            ModuleNotFoundError: If albumentations or tables is not installed.
            ValueError: If configured arrays have no sample axis or different
                numbers of samples. Unselected metadata arrays are ignored.
        """
        try:
            import_module("albumentations")
        except ModuleNotFoundError as error:
            if error.name != "albumentations":
                raise
            raise ModuleNotFoundError(
                "The albumentations lib is needed for TrainDatasetH5. "
                "Install with `pip install albumentations`"
            ) from error

        try:
            tb = import_module("tables")
        except ModuleNotFoundError as error:
            if error.name != "tables":
                raise
            raise ModuleNotFoundError(
                "The tables lib is needed for TrainDatasetH5. "
                "Install with `pip install tables`"
            ) from error

        self.path = path
        self.img_key = img_key
        self.inst_keys = inst_keys
        self.mask_keys = mask_keys
        self.keys = [img_key] + list(mask_keys) + list(inst_keys)
        self.transforms = transforms
        self.inst_transforms = inst_transforms

        with tb.open_file(path, "r") as h5:
            image = h5.root[img_key]
            if not image.shape:
                raise ValueError(f"'{img_key}' must have a sample axis")
            self.n_items = image.shape[0]
            for key in self.keys:
                array = h5.root[key]
                if not array.shape or array.shape[0] != self.n_items:
                    raise ValueError(
                        f"'{key}' must have {self.n_items} samples matching '{img_key}'"
                    )

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return self.n_items

    def __getitem__(self, ix: int) -> dict[str, torch.Tensor]:
        """Read aligned image, integer masks and float32 auxiliary targets.

        Images and auxiliary maps are CHW tensors; integer masks are HW tensors.
        Auxiliary keys retain the format ``{instance_key}_{transform_name}``.
        """
        data = FileHandler.read_h5(self.path, ix, keys=self.keys)

        # get instance transform kwargs
        inst_kws = {k: data[k] for k in self.inst_keys}

        # apply instance transforms
        aux = self.inst_transforms(**inst_kws)

        # append integer masks and instance transformed masks
        masks = [data[k][..., np.newaxis] for k in self.mask_keys] + aux

        # number of channels per non image data
        mask_chls = [m.shape[2] for m in masks]

        # concatenate all masks + inst transforms
        masks = np.concatenate(masks, axis=-1)
        tr = self.transforms(image=data[self.img_key], masks=[masks])

        image = to_tensor(tr["image"])
        masks = to_tensor(tr["masks"][0])
        masks = torch.split(masks, mask_chls, dim=0)

        integer_masks = {
            n: masks[i].squeeze().long() for i, n in enumerate(self.mask_keys)
        }
        aux_keys = [
            f"{key}_{name}"
            for name in self.inst_transforms.names
            for key in self.inst_keys
        ]
        inst_transformed_masks = {
            name: masks[len(integer_masks) + i].float()
            for i, name in enumerate(aux_keys)
        }

        out = {self.img_key: image.float(), **inst_transformed_masks, **integer_masks}

        return out
