from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cellseg_models_pytorch.utils import FileHandler, to_tensor

if TYPE_CHECKING:
    import albumentations as A

    from cellseg_models_pytorch.transforms.albu_transforms import ApplyEach

__all__ = ["TrainDatasetFolder"]

ALLOWED_KEYS = ("inst", "type", "cyto_inst", "cyto_type", "sem")


class TrainDatasetFolder(Dataset):
    """Load aligned RGB images and MATLAB label masks for training.

    Image and label files are sorted separately and paired by position, not matched
    by filename. Keep both folders in the same order with no unrelated files.
    Spatial augmentation runs jointly before auxiliary targets are generated.
    """

    def __init__(
        self,
        path_img: str,
        path_label: str,
        mask_keys: Tuple[str, ...],
        transforms: A.Compose,
        inst_transforms: ApplyEach,
        map_out_keys: Dict[str, str] = None,
    ) -> None:
        """Choose folders, labels and training transforms.

        Args:
            path_img: Folder of RGB images supported by the image reader.
            path_label: Folder of .mat files, one per image in the same sorted order.
            mask_keys: Label arrays to read: inst, type, cyto_inst, cyto_type or sem.
                Must include inst. Masks use height, width axes; instance 0 is background
                and positive labels identify objects. Class meanings come from your data.
            transforms: Albumentations Compose receiving image and a list of masks.
                Include the normalization expected by the model.
            inst_transforms: ApplyEach with as_list=True, deriving targets from the
                augmented instance masks. Vector targets are converted to CHW tensors.
            map_out_keys: Optional dictionary selecting and renaming output entries.
                Only listed existing keys are retained. To keep the image, include
                image explicitly. Auxiliary keys are transform_name_instance_key,
                such as cellpose_inst.

        Raises:
            ModuleNotFoundError: Albumentations is unavailable.
            ValueError: mask_keys contains unsupported keys or omits inst. Length
                access also rejects unequal image/label file counts.
        """
        try:
            import_module("albumentations")
        except ModuleNotFoundError as error:
            if error.name != "albumentations":
                raise
            raise ModuleNotFoundError(
                "The albumentations lib is needed for TrainDatasetFolder. "
                "Install with `pip install albumentations`"
            ) from error

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
        """Return the paired sample count.

        Raises:
            ValueError: Image and label folders contain different numbers of files.
        """
        if len(self.img_files) != len(self.label_files):
            raise ValueError("The number of images and labels must be the same.")
        return len(self.img_files)

    def __getitem__(self, ix: int) -> dict[str, torch.Tensor]:
        """Read one augmented training sample.

        Args:
            ix: Position in the separately sorted image and label lists.

        Returns:
            Dictionary of CPU tensors: image is CHW float32, selected labels are HW,
            and generated targets are CHW (for example cellpose_inst). Use
            map_out_keys to select or rename entries. H is height, W width and
            C channels.

            Label and target dtypes depend on the transforms. Convert class labels
            to long and flow/distance targets to float before computing losses.
            A spatial dimension of size one may be removed from labels; use
            patches with both spatial dimensions greater than one.
        """
        img = {"image": FileHandler.read_img(self.img_files[ix])}
        masks = FileHandler.read_mat(self.label_files[ix])
        masks = {k: v for k, v in masks.items() if k in self.mask_keys}
        data = {**img, **masks}

        # Augment labels first; derive vector targets from their new geometry.
        tr = self.transforms(
            image=data["image"], masks=[data[key] for key in self.mask_keys]
        )
        transformed = dict(zip(self.mask_keys, tr["masks"]))
        aux = self.inst_transforms(
            **{key: transformed[key] for key in self.inst_in_keys}
        )
        masks = [transformed[key][..., np.newaxis] for key in self.mask_keys] + aux
        mask_chls = [mask.shape[2] for mask in masks]

        image = to_tensor(tr["image"])
        masks = torch.split(to_tensor(np.concatenate(masks, axis=-1)), mask_chls, dim=0)

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
