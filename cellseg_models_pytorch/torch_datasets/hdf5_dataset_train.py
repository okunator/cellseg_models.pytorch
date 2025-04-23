from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.transforms.albu_transforms import ApplyEach
from cellseg_models_pytorch.utils import FileHandler, to_tensor

try:
    import albumentations as A

    has_albu = True
except ModuleNotFoundError:
    has_albu = False

try:
    import tables as tb

    has_tb = True
except ModuleNotFoundError:
    has_tb = False

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

        Parameters:
            path (str):
                Path to the h5 file.
            img_key (str):
                Key for the image data in the h5 file.
            inst_keys (Tuple[str, ...]):
                Key for the instance data in the h5 file. This will be transformed
            mask_keys (Tuple[str, ...]):
                Keys for the semantic masks in the h5 file.
            transforms (A.Compose):
                Albumentations compose object for image and mask transforms.
            inst_transforms (ApplyEach):
                ApplyEach object for instance transforms.

        Raises:
            ModuleNotFoundError: If albumentations or tables is not installed.
            ModuleNotFoundError: If tables is not installed.
        """
        if not has_albu:
            raise ModuleNotFoundError(
                "The albumentations lib is needed for TrainDatasetH5. "
                "Install with `pip install albumentations`"
            )

        if not has_tb:
            raise ModuleNotFoundError(
                "The tables lib is needed for TrainDatasetH5. "
                "Install with `pip install tables`"
            )

        self.path = path
        self.img_key = img_key
        self.inst_keys = inst_keys
        self.mask_keys = mask_keys
        self.keys = [img_key] + list(mask_keys) + list(inst_keys)
        self.transforms = transforms
        self.inst_transforms = inst_transforms

        with tb.open_file(path, "r") as h5:
            for array in h5.walk_nodes("/", classname="Array"):
                self.n_items = len(array)
                break

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return self.n_items

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
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
        inst_transformed_masks = {
            f"{n}_{tr_n}": masks[len(integer_masks) + i].float()
            for n in self.inst_keys
            for i, tr_n in enumerate(self.inst_transforms.names)
        }

        out = {self.img_key: image.float(), **inst_transformed_masks, **integer_masks}

        return out
