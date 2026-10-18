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
    """Load images and aligned masks from an existing HDF5 file for training.

    Requires PyTables and Albumentations. Configured nodes must be arrays with
    matching first-axis sample counts. Other metadata nodes are ignored.
    """

    def __init__(
        self,
        path: str,
        img_key: str,
        inst_keys: Tuple[str, ...],
        mask_keys: Tuple[str, ...],
        transforms: A.Compose,
        inst_transforms: ApplyEach,
    ) -> None:
        """Choose HDF5 arrays and training transforms.

        Args:
            path: Existing HDF5 file readable by PyTables.
            img_key: Root image node name, containing (N, H, W, C) RGB images.
            inst_keys: Root instance nodes, each (N, H, W), used to generate targets.
                Zero is background; positive labels identify individual objects.
            mask_keys: Root label nodes retained in the returned sample, each
                (N, H, W). Include an instance node here if you also need its labels.
            transforms: Albumentations Compose for joint image/mask augmentation
                and checkpoint-specific image normalization.
            inst_transforms: ApplyEach with as_list=True, applied after augmentation.

        Raises:
            ModuleNotFoundError: Albumentations or PyTables is unavailable.
            ValueError: A selected array has no sample axis or has a different count
                from img_key. Missing files/nodes raise the underlying reader error.
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
        """Return the shared sample count of the selected arrays."""
        return self.n_items

    def __getitem__(self, ix: int) -> dict[str, torch.Tensor]:
        """Read one aligned, augmented HDF5 training sample.

        Args:
            ix: Index on the arrays' shared sample axis.

        Returns:
            CPU tensor dictionary. The image uses img_key, CHW axes and float32 dtype.
            Labels use mask_keys and int64 dtype; HW labels are squeezed, including
            any size-one spatial axes. Auxiliary targets use CHW float32 tensors
            named instance_key_transform_name, such as inst_cellpose. This naming
            order differs from TrainDatasetFolder. N is sample count, H height,
            W width and C channels. Values follow the configured transforms.
        """
        data = FileHandler.read_h5(self.path, ix, keys=self.keys)

        # Augment labels first; derive vector targets from their new geometry.
        keys = list(dict.fromkeys((*self.mask_keys, *self.inst_keys)))
        tr = self.transforms(
            image=data[self.img_key], masks=[data[key] for key in keys]
        )
        transformed = dict(zip(keys, tr["masks"]))
        aux = self.inst_transforms(**{key: transformed[key] for key in self.inst_keys})
        masks = [transformed[key][..., np.newaxis] for key in self.mask_keys] + aux
        mask_chls = [mask.shape[2] for mask in masks]

        image = to_tensor(tr["image"])
        masks = torch.split(to_tensor(np.concatenate(masks, axis=-1)), mask_chls, dim=0)

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
