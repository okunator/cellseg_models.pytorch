from pathlib import Path
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


__all__ = ["TrainDatasetFolder"]

ALLOWED_KEYS = ("inst", "type", "cyto_inst", "cyto_type", "sem")


class TrainDatasetFolder(Dataset):
    def __init__(
        self,
        path_img: str,
        path_label: str,
        mask_keys: Tuple[str, ...],
        transforms: A.Compose,
        inst_transforms: ApplyEach,
        map_out_keys: Dict[str, str] = None,
    ) -> None:
        """Folder train dataset for cell/panoptic segmentation models.

        Parameters:
            path_img (str):
                Path to the image folder.
            path_label (str):
                Path to the label folder.
            mask_keys (Tuple[str, ...]):
                Tuple of keys to be read from the .mat file.
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

        if not all(k in ALLOWED_KEYS for k in mask_keys):
            raise ValueError(
                f"Invalid keys. Allowed keys are {ALLOWED_KEYS}, got {mask_keys}"
            )

        if "inst" not in mask_keys:
            raise ValueError("'inst' key must be present in keys")

        self.path_img = Path(path_img)
        self.path_label = Path(path_label)
        self.img_files = sorted(list(self.path_img.glob("*")))
        self.label_files = sorted(list(self.path_label.glob("*")))
        self.mask_keys = mask_keys
        self.inst_in_keys = [k for k in mask_keys if "inst" in k]
        self.inst_out_keys = [
            f"{name}_{key}"
            for name in inst_transforms.names
            for key in self.inst_in_keys
        ]
        self.transforms = transforms
        self.inst_transforms = inst_transforms
        self.map_out_keys = map_out_keys

    def __len__(self) -> int:
        """Return the number of items in the db."""
        if len(self.img_files) != len(self.label_files):
            raise ValueError("The number of images and labels must be the same.")
        return len(self.img_files)

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
        img = {"image": FileHandler.read_img(self.img_files[ix])}
        masks = FileHandler.read_mat(self.label_files[ix])
        masks = {k: v for k, v in masks.items() if k in self.mask_keys}
        data = {**img, **masks}

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
            # k: masks[i].squeeze().long() for i, k in enumerate(self.mask_keys)
            k: masks[i].squeeze()
            for i, k in enumerate(self.mask_keys)
        }
        inst_transformed_masks = {
            # f"{n}": masks[len(integer_masks) + i].float()
            n: masks[len(integer_masks) + i]
            for i, n in enumerate(self.inst_out_keys)
        }

        out = {"image": image.float(), **integer_masks, **inst_transformed_masks}

        if self.map_out_keys is not None:
            new_out = {}
            for in_key, out_key in self.map_out_keys.items():
                if in_key in out:
                    new_out[out_key] = out.pop(in_key)
            out = new_out

        return out
