from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset

from ..transforms import (
    IMG_TRANSFORMS,
    INST_TRANSFORMS,
    NORM_TRANSFORMS,
    apply_each,
    compose,
    to_tensorv3,
)

try:
    import tables as tb
except Exception:
    raise ImportError(
        "`pytables` needed for this class. Install with: `pip install tables`"
    )


__all__ = ["SegmentationHDF5Dataset"]


class SegmentationHDF5Dataset(Dataset):
    def __init__(
        self,
        path: str,
        img_transforms: List[str],
        inst_transforms: List[str],
        normalization: str = None,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        return_weight: bool = False,
        **kwargs,
    ) -> None:
        """Create a dataset class that reads images/patches from a hdf5 database.

        This class requires a structured HDF5 database. You can create such a database
        with `HDF5Writer`.

        Parameters
        ----------
            path : str
                Path to the hdf5 database.
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
        """
        self.path = Path(path)

        if self.path.suffix not in (".h5", ".hdf5"):
            raise ValueError(
                f"""The input path has to be a hdf5 db. Got suffix: {self.path.suffix}
                Allowed suffices: {(".h5", ".hdf5")}"""
            )

        allowed = list(IMG_TRANSFORMS.keys())
        if not all([tr in allowed for tr in allowed]):
            raise ValueError(
                f"Wrong img transformation. Got: {img_transforms}. Allowed: {allowed}."
            )
        allowed = list(NORM_TRANSFORMS.keys())
        if not all([tr in allowed for tr in allowed]):
            raise ValueError(
                f"Wrong norm transformation. Got: {img_transforms}. Allowed: {allowed}."
            )

        allowed = list(INST_TRANSFORMS.keys())
        if not all([tr in allowed for tr in allowed]):
            raise ValueError(
                f"Wrong inst transformation. Got: {img_transforms}. Allowed: {allowed}."
            )

        # Return masks
        self.return_inst = return_inst
        self.return_type = return_type
        self.return_sem = return_sem

        # Set transformations
        img_transforms = [IMG_TRANSFORMS[tr](**kwargs) for tr in img_transforms]
        if normalization is not None:
            img_transforms.append(NORM_TRANSFORMS[normalization]())

        inst_transforms = [INST_TRANSFORMS[tr]() for tr in inst_transforms]
        if return_inst:
            inst_transforms.append(INST_TRANSFORMS["binarize"]())

        if return_weight:
            inst_transforms.append(INST_TRANSFORMS["edgeweight"]())

        self.img_transforms = compose(img_transforms)
        self.inst_transforms = apply_each(inst_transforms)
        self.to_tensor = to_tensorv3()

        # n imgs in the db.
        with tb.open_file(self.path) as h5:
            self.n_items = h5.root._v_attrs["n_items"]

    @staticmethod
    def read_h5_patch(
        path: str,
        ix: int,
        return_type: bool = True,
        return_sem: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Read img & mask patches at index `ix`.

        Parameters
        ----------
            path : str
                Path to the hdf5 database.
            ix : int
                Index for the hdf5 db-arrays.
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
            IOError: If a mask that does not exist in the db is being read.
        """
        with tb.open_file(path, "r") as h5:
            out = OrderedDict()
            out["image"] = h5.root.imgs[ix, ...]

            try:
                out["inst"] = h5.root.insts[ix, ...]
            except Exception:
                raise IOError(
                    "The HDF5 database does not contain instance labelled masks."
                )

            if return_type:
                try:
                    out["type"] = h5.root.types[ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain type masks.")

            if return_sem:
                try:
                    out["sem"] = h5.root.areas[ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain semantic masks.")

            return out

    def __len__(self) -> int:
        """Return the number of items in the db."""
        return self.n_items

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
        inputs = self.read_h5_patch(self.path, ix, self.return_type, self.return_sem)

        # wrangle inputs to albumentations format
        mask_names = [key for key in inputs.keys() if key != "image"]
        masks = [arr for key, arr in inputs.items() if key != "image"]
        data = dict(image=inputs["image"], masks=masks)

        # transform + convert to tensor
        aug = self.img_transforms(**data)
        aux = self.inst_transforms(image=aug["image"], inst_map=aug["masks"][0])
        data = self.to_tensor(image=aug["image"], masks=aug["masks"], aux=aux)

        # wrangle data to return format
        out = dict(image=data["image"])
        for m, n in zip(data["masks"], mask_names):
            out[f"{n}"] = m

        for n, aux_map in aux.items():
            out[f"{n}"] = aux_map

        # remove redundant target (not needed in downstream).
        if self.return_inst:
            out["inst"] = out["binary"]
            del out["binary"]
        else:
            del out["inst"]

        return out
