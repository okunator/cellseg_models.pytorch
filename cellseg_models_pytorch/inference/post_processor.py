from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from skimage.util import img_as_ubyte

from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP
from cellseg_models_pytorch.utils import (
    FileHandler,
    fill_holes_semantic,
    majority_vote_parallel,
    majority_vote_sequential,
    med_filt_parallel,
    med_filt_sequential,
    remove_debris_semantic,
)

__all__ = ["PostProcessor"]


INST_ARGMAX = {
    "stardist": False,
    "stardist_orig": False,
    "cellpose": True,
    "cellpose_old": True,
    "omnipose": True,
    "hovernet": True,
    "drfns": True,
    "dcan": False,
    "dran": False,
}

ALLOWED_FORMATS = (".mat", ".geojson", ".feather", ".parquet")


class PostProcessor:
    def __init__(
        self,
        method: str,
        inst_key: str,
        aux_key: str,
        type_post_proc: Callable = None,
        cyto_post_proc: Callable = None,
        sem_post_proc: Callable = None,
    ) -> None:
        """A class for post-processing of all the different types of model outputs.

        Note:
            The post-processing methods are not meant to be used directly. The class
            provides a pipeline for post-processing the instance, type, and semantic
            segmentation maps.

        Parameters:
            method (str):
                The post-processing method for the instance segmentation mask.
            inst_key (str):
                The key/name of the model instance prediction output that is used for the
                  instance segmentation post-processing pipeline.
            aux_key (Tuple[str, ...]):
                The key/name of the model auxiliary output that is used for the instance
                segmentation post-processing pipeline.
            type_post_proc (Callable, default=None):
                A post-processing function for the type maps. If not None, overrides the
                default.
            cyto_post_proc (Callable, default=None):
                A post-processing function for the cytoplasm segmentation maps. If not
                None, overrides the default.
            sem_post_proc (Callable, default=None):
                A post-processing function for the semantic segmentation maps. If not
                None, overrides the default.

        Raises:
            ValueError: If the post-processing method is not allowed.
        """
        allowed = list(POSTPROC_LOOKUP.keys())
        if method not in allowed:
            raise ValueError(
                f"Illegal post proc method. Got {method}. Allowed: {allowed}"
            )

        self.postproc_method = POSTPROC_LOOKUP[method]
        self.use_argmax = INST_ARGMAX[method]
        self.inst_key = inst_key
        self.aux_key = aux_key
        self.type_post_proc = type_post_proc
        self.cyto_post_proc = cyto_post_proc
        self.sem_post_proc = sem_post_proc

    def get_inst_map(
        self, prob_map: np.ndarray, aux_map: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Run the instance labelling post processing."""
        if self.use_argmax:
            inst = np.argmax(prob_map, axis=0)
        else:
            inst = prob_map.squeeze()

        return self.postproc_method(inst, aux_map, **kwargs)

    def get_type_map(
        self,
        prob_map: np.ndarray,
        inst_map: np.ndarray,
        parallel: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Run the type map post-processing. Majority voting for each instance."""
        type_map = np.argmax(prob_map, axis=0)

        if self.type_post_proc is not None:
            type_map = self.type_post_proc(type_map, inst_map, **kwargs)
        else:
            if parallel:
                type_map = majority_vote_parallel(type_map, inst_map)
            else:
                type_map = majority_vote_sequential(type_map, inst_map)

        return type_map.astype("i4")

    def get_sem_map(
        self,
        prob_map: np.ndarray,
        parallel: bool = False,
        kernel_width: int = 15,
        **kwargs,
    ) -> np.ndarray:
        """Run the semantic segmentation post-processing."""
        sem_map = img_as_ubyte(prob_map)

        if self.sem_post_proc is not None:
            sem = self.sem_post_proc(sem_map, **kwargs)
        else:
            if parallel:
                sem = med_filt_parallel(
                    sem_map, kernel_size=(kernel_width, kernel_width)
                )
            else:
                sem = med_filt_sequential(sem_map, kernel_width)

            sem = np.argmax(sem, axis=0)
            sem = remove_debris_semantic(sem)
            sem = fill_holes_semantic(sem)

        return sem.astype("i4")

    def post_proc_pipeline(
        self,
        inst_map: np.ndarray,
        aux_map: np.ndarray,
        type_map: np.ndarray = None,
        sem_map: np.ndarray = None,
        cyto_inst_map: np.ndarray = None,
        cyto_type_map: np.ndarray = None,
        *,
        dst: str = None,
        coords: Tuple[int, int, int, int] = None,
        save_kwargs: Dict = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run post-processing for all the model outputs.

        Parameters:
            inst_map (np.ndarray):
                The model instance segmentation map. Shape (H, W).
            aux_map (np.ndarray):
                The model auxiliary output map. Shape (C, H, W).
            type_map (np.ndarray, default=None):
                The model type map. Shape (C, H, W).
            sem_map (np.ndarray, default=None):
                The model semantic segmentation map. Shape (C, H, W).
            cyto_map (np.ndarray, default=None):
                The model cytoplasm segmentation map. Shape (C, H, W).
            dst (str, default=None):
                The path to save the post-processed output. If None,
                the output is not saved. Allowed formats: .mat, .geojson, .feather,
                .parquet.
            save_kwargs (Dict[str, Any], default=None):
                Keyword arguments for saving the post-processed predictions.
                See `FileHandler.to_mat`, `FileHandler.to_gson`, `FileHandler.to_h5`
                for more details.
            **kwargs:
                Arbitrary keyword arguments that can be used for any of the private
                post-processing functions of this class.

        Returns:
            Dict[str, np.ndarray]:
                Final output names mapped to the final post-processed outputs.
                E.g. {"sem": np.ndarray, "type": np.ndarray, "inst": np.ndarray}
        """
        save_kwargs = save_kwargs or {}

        if dst is not None:
            dst = save_path = Path(dst)
            suff = save_path.suffix

            if suff not in ALLOWED_FORMATS:
                raise ValueError(
                    f"Illegal `dst` format. Got {save_path.as_posix()}. "
                    f"Allowed: {ALLOWED_FORMATS}."
                )

        res = {}
        if sem_map is not None:
            res["sem"] = self.get_sem_map(sem_map, **kwargs)

        if cyto_inst_map is not None:
            pass

        if cyto_type_map is not None:
            pass

        # inst map should typically be a binary bg, fg mask. If the type map is not
        # provided, and `inst_map` contains more than two classes, then we can assume
        # that `inst_map` == `type_map` and do also type post-processing.
        if np.unique(inst_map).size > 2:
            type_map = inst_map

        res["inst"] = self.get_inst_map(inst_map, aux_map, **kwargs)

        if type_map is not None:
            res["type"] = self.get_type_map(type_map, res["inst"], **kwargs)
            res["inst"] *= res["type"] > 0

        if dst is not None:
            if suff in (".geojson", ".feather", ".parquet"):
                FileHandler.to_gson(res, dst, coords, **save_kwargs)
            elif suff == ".mat":
                FileHandler.to_mat(res, dst, coords, **save_kwargs)
        else:
            return res
