from pathlib import Path
from typing import Union

import numpy as np
import scipy.io

from .mask_utils import bounding_box, get_inst_centroid, get_inst_types

__all__ = ["mask2mat"]


def mask2mat(
    fname: Union[str, Path],
    save_dir: Union[str, Path],
    inst: np.ndarray,
    type: np.ndarray = None,
    sem: np.ndarray = None,
) -> None:
    """Convert one set of segmentation model output masks into a .mat file.

    Keys: "inst_map", "inst_type", "inst_centroid", "inst_bbox"
    Optional: "type_map", "sem_map",

    Parameters
    ----------
        fname : str or Path
            File name for the .mat file.
        save_dir : str or Path
            directory where the .mat files are saved.
        inst : np.ndarray
            Instance labelled instance segmentation mask from the segmentation model.
        sem : np.ndarray, optional
            Semantic segmentation mask from the segmentation model.
        type : np.ndarray, optional
            Cell type labelled semantic segmentation mask from the segmentation model.
    """
    save_dir = Path(save_dir)
    fname = Path(fname).with_suffix(".mat").name
    fn_mask = Path(save_dir / fname)

    if not Path(save_dir).exists():
        raise ValueError(f"`save_dir`: {save_dir} does not exist.")

    centroids = get_inst_centroid(inst)
    inst_ids = list(np.unique(inst)[1:])
    bboxes = np.array(
        [bounding_box(np.array(inst == id_, np.uint8)) for id_ in inst_ids]
    )

    mdict = {"inst_map": inst, "inst_centroids": centroids, "inst_bbox": bboxes}

    if type is not None:
        inst_types = get_inst_types(inst, type)
        mdict["inst_type"] = inst_types
        mdict["type_map"] = type

    if sem is not None:
        mdict["sem_map"] = sem

    scipy.io.savemat(file_name=fn_mask.as_posix(), mdict=mdict)
