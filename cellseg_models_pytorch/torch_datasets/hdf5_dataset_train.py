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

ALLOWED_KEYS = ("image", "inst", "type", "cyto_inst", "cyto_type", "sem")


class TrainDatasetH5(Dataset):
    def __init__(
        self,
        path: str,
        input_keys: Tuple[str, ...],
        transforms: A.Compose,
        inst_transforms: ApplyEach,
        map_out_keys: Dict[str, str] = None,
    ) -> None:
        """HDF5 train dataset for cell/panoptic segmentation models.

        Parameters:
            path (str):
                Path to the h5 file.
            input_keys (Tuple[str, ...]):
                Tuple of keys to be read from the h5 file.
            transforms (A.Compose):
                Albumentations compose object for image and mask transforms.
            inst_transforms (ApplyEach):
                ApplyEach object for instance transforms.
            map_out_keys (Dict[str, str], default=None):
                A dictionary to map the default output keys to new output keys. .
                Useful if you want to match the output keys with model output keys.
                e.g. {"inst": "decoder1-inst", "inst-cellpose": decoder2-cellpose}.
                The default output keys are any of 'image', 'inst', 'type', 'cyto_inst',
                'cyto_type', 'sem' & inst-{transform.name}, cyto_inst-{transform.name}.

        Raises:
            ModuleNotFoundError: If albumentations or tables is not installed.
            ModuleNotFoundError: If tables is not installed.
            ValueError: If invalid keys are provided.
            ValueError: If 'image' key is not present in input_keys.
            ValueError: If 'inst' key is not present in input_keys.
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

        if not all(k in ALLOWED_KEYS for k in input_keys):
            raise ValueError(
                f"Invalid keys. Allowed keys are {ALLOWED_KEYS}, got {input_keys}"
            )

        if "image" not in input_keys:
            raise ValueError("'image' key must be present in keys")

        if "inst" not in input_keys:
            raise ValueError("'inst' key must be present in keys")

        self.path = path
        self.keys = input_keys
        self.mask_keys = [k for k in input_keys if k != "image"]
        self.inst_in_keys = [k for k in input_keys if "inst" in k]
        self.inst_out_keys = [
            f"{key}-{name}"
            for name in inst_transforms.names
            for key in self.inst_in_keys
        ]
        self.transforms = transforms
        self.inst_transforms = inst_transforms
        self.map_out_keys = map_out_keys

        with tb.open_file(path, "r") as h5:
            self.n_items = len(h5.root["fname"][:])

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return self.n_items

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
        data = FileHandler.read_h5(self.path, ix, keys=self.keys)

        # get instance transform kwargs
        inst_kws = {
            k: data[k] for k in self.inst_in_keys if data.get(k, None) is not None
        }

        # apply instance transforms
        aux = self.inst_transforms(**inst_kws)

        # append integer masks and instance transformed masks
        masks = [d[..., np.newaxis] for k, d in data.items() if k != "image"] + aux

        # number of channels per non image data
        mask_chls = [m.shape[2] for m in masks]

        # concatenate all masks + inst transforms
        masks = np.concatenate(masks, axis=-1)

        tr = self.transforms(image=data["image"], masks=[masks])

        image = to_tensor(tr["image"])
        masks = to_tensor(tr["masks"][0])
        masks = torch.split(masks, mask_chls, dim=0)

        integer_masks = {
            n: masks[i].squeeze().long()
            for i, n in enumerate(self.mask_keys)
            # n: masks[i].squeeze()
            # for i, n in enumerate(self.mask_keys)
        }
        inst_transformed_masks = {
            # n: masks[len(integer_masks) + i]
            # for i, n in enumerate(self.inst_out_keys)
            n: masks[len(integer_masks) + i].float()
            for i, n in enumerate(self.inst_out_keys)
        }

        # out = {"image": image.float(), **integer_masks, **inst_transformed_masks}
        out = {"image": image.float(), **integer_masks, **inst_transformed_masks}

        if self.map_out_keys is not None:
            new_out = {}
            for in_key, out_key in self.map_out_keys.items():
                if in_key in out:
                    new_out[out_key] = out.pop(in_key)
            out = new_out

        return out
