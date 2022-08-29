from pathlib import Path
from typing import Union

import cv2
import numpy as np
import scipy.io as sio

from .mask_utils import bounding_box, fix_duplicates, get_inst_centroid, get_inst_types


class FileHandler:
    """Class for handling file I/O."""

    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        """Read an image & convert from bgr to rgb. (cv2 reads imgs in bgr).

        Parameters
        ----------
            path : str or Path
                Path to the image file.

        Returns
        -------
            np.ndarray:
                The image. Shape (H, W, 3).
        """
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def write_img(path: Union[str, Path], img: np.ndarray) -> None:
        """Write an image.

        Parameters
        ----------
            path : str or Path
                Path to the image file.
            img : np.ndarray
                The image to be written.

        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def read_mask(
        path: Union[str, Path],
        key: str = "inst_map",
        retype: bool = True,
        return_all: bool = False,
    ) -> Union[np.ndarray, None]:
        """Read a mask from a .mat file.

        If a mask is not found, return None

        Parameters
        ----------
            path : str or Path
                Path to the image file.
            key : str, default="inst_map"
                Name/key of the mask type that is being read from .mat
            retype : bool, default=True
                Convert the matrix type.
            return_all : bool, default=False
                Return the whole dict. Overrides the `key` arg.

        Returns
        -------
            np.ndarray or None:
                The mask indice matrix. Shape (H, W)

        Raises
        ------
            ValueError: If an illegal key is given.
        """
        dtypes = {
            "inst_map": "int32",
            "type_map": "int32",
            "sem_map": "int32",
            "inst_centroid": "float64",
            "inst_type": "int32",
        }

        path = Path(path)
        if not path.exists():
            raise ValueError(f"{path} not found")

        try:
            mask = sio.loadmat(path.as_posix())
        except Exception:
            mask = None

        if not return_all:
            allowed = ("inst_map", "type_map", "inst_centroid", "inst_type", "sem_map")
            if key not in allowed:
                raise ValueError(f"Illegal key given. Got {key}. Allowed: {allowed}")

            try:
                mask = mask[key]
                if retype:
                    mask = mask.astype(dtypes[key])
            except Exception:
                mask = None

        return mask

    @staticmethod
    def write_mask(
        path: Union[str, Path],
        inst_map: np.ndarray,
        type_map: np.ndarray = None,
        sem_map: np.ndarray = None,
    ) -> None:
        """
        Write multiple masks to .mat file.

        Parameters
        ----------
            path : str or Path
                Path to the .mat file.
            inst_map : np.ndarray
                The inst_map to be written.
            type_map : np.ndarray, optional
                The inst_map to be written.
            sem_map : np.ndarray, optional
                The inst_map to be written.
        """
        path = Path(path)
        if not path.suffix == ".mat":
            raise ValueError(f"File suffix needs to be '.mat'. Got {path.suffix}.")

        inst_map = fix_duplicates(inst_map)
        centroids = get_inst_centroid(inst_map)
        inst_types = get_inst_types(inst_map, type_map)
        inst_ids = list(np.unique(inst_map)[1:])
        bboxes = np.array(
            [bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids]
        )

        res = {
            "inst_map": inst_map,
            "inst_type": inst_types,
            "inst_centroid": centroids,
            "inst_bbox": bboxes,
        }

        if type_map is not None:
            res["type_map"] = type_map

        if sem_map is not None:
            res["sem_map"] = type_map

        sio.savemat(
            file_name=path,
            mdict=res,
        )
