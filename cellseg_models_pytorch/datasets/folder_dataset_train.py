from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..utils import FileHandler
from ._base_dataset import TrainDatasetBase

IMG_SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")
MASK_SUFFIXES = (".mat",)

__all__ = ["SegmentationFolderDataset"]


class SegmentationFolderDataset(TrainDatasetBase):
    def __init__(
        self,
        path: str,
        mask_path: str,
        img_transforms: List[str],
        inst_transforms: List[str],
        normalization: str = None,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        return_weight: bool = False,
        **kwargs,
    ) -> None:
        """Create a dataset class that reads images/patches from a folder.

        Parameters
        ----------
            path : str
                Path to the folder containing the images.
            mask_path : str
                Path to the folder containing the corresponding masks (.mat files).
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            return_inst : bool, default=True
                If True, returns a binarized instance mask. (If the db contains these.)
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)
            return_weight : bool, default=False
                Include a nuclear border weight map in the output.

        Raises
        ------
            ValueError if there are issues with the given paths or files.
        """
        super().__init__(
            img_transforms=img_transforms,
            inst_transforms=inst_transforms,
            normalization=normalization,
            return_inst=return_inst,
            return_type=return_type,
            return_sem=return_sem,
            return_weight=return_weight,
            **kwargs,
        )

        self.path = Path(path)
        self.mask_path = Path(mask_path)

        if not self.path.exists():
            raise ValueError(f"folder: {path} does not exist")

        if not self.path.is_dir():
            raise ValueError(f"path: {path} is not a folder")

        if not all([f.suffix in IMG_SUFFIXES for f in self.path.iterdir()]):
            raise ValueError(
                f"files formats in given folder need to be in {IMG_SUFFIXES}"
            )

        if not self.mask_path.exists():
            raise ValueError(f"folder: {self.mask_path} does not exist")

        if not self.mask_path.is_dir():
            raise ValueError(f"path: {self.mask_path} is not a folder")

        if not all([f.suffix in MASK_SUFFIXES for f in self.mask_path.iterdir()]):
            raise ValueError(
                f"files formats in given folder need to be in {MASK_SUFFIXES}"
            )

        self.fnames_imgs = sorted(self.path.glob("*"))
        self.fnames_masks = sorted(self.mask_path.glob("*"))
        if len(self.fnames_imgs) != len(self.fnames_masks):
            raise ValueError(
                f"Found different number of files in {self.path.as_posix()} and "
                f"{self.mask_path.as_posix()}."
            )

    def read_img_mask(
        self,
        ix: int,
        return_type: bool = True,
        return_sem: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Read img & mask patches at index `ix`.

        Parameters
        ----------
            ix : int
                Index for the img in the folder.
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)

        Returns
        -------
            OrderedDict[str, np.ndarray]:
                A Dict of numpy matrices. Img shape: (H, W, 3), mask shapes: (H, W).
                keys of the dict are: "im", "inst", "type", "sem"

        Raises
        ------
            KeyError: If a mask that does not exist in a given .mat file.
        """
        out = OrderedDict()
        out["image"] = FileHandler.read_img(self.fnames_imgs[ix])
        masks = FileHandler.read_mask(self.fnames_masks[ix], return_all=True)

        try:
            out["inst"] = masks["inst_map"]
        except KeyError:
            raise KeyError(
                f"The file {self.fnames_masks[ix]} does not contain key `inst_map`."
            )

        if return_type:
            try:
                out["type"] = masks["type_map"]
            except KeyError:
                raise KeyError(
                    f"The file {self.fnames_masks[ix]} does not contain key `type_map`."
                )

        if return_sem:
            try:
                out["sem"] = masks["sem_map"]
            except KeyError:
                raise KeyError(
                    f"The file {self.fnames_masks[ix]} does not contain key `sem_map`."
                )

        return out

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return len(self.fnames_imgs)

    def __getitem__(self, ix: int) -> Dict[str, np.ndarray]:
        """Get item.

        Parameters
        ----------
            ix : int
                An index for the iterable dataset.

        Returns
        -------
            Dict[str, np.ndarray]:
                A dictionary containing all the augmented data patches.
                Keys are: "im", "inst", "type", "sem". Image shape: (B, 3, H, W).
                Mask shapes: (B, C_mask, H, W).

        """
        return self._getitem(ix, self.read_img_mask)
