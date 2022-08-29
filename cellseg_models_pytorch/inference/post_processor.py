from typing import Dict, List

import numpy as np
from pathos.multiprocessing import ThreadPool as Pool
from skimage.filters import rank
from skimage.morphology import closing, disk, opening
from skimage.util import img_as_ubyte
from tqdm import tqdm

from ..postproc import POSTPROC_LOOKUP
from ..utils import binarize, fill_holes_semantic, remove_debris_semantic

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


class PostProcessor:
    def __init__(
        self, instance_postproc: str, inst_key: str, aux_key: str, **kwargs
    ) -> None:
        """Multi-threaded post-processing.

        Parameters
        ----------
            instance_postproc : str
                The post-processing method for the instance segmentation mask.
            inst_key : str
                The key/name of the model instance prediction output that is used
                for the instance segmentation post-processing pipeline.
            aux_key : Tuple[str, ...]:
                The key/name of the model auxilliary output that is used for the
                instance segmentation post-processing pipeline.
            **kwargs
                Arbitrary keyword arguments that can be used for any of the private
                post-processing functions of this class.
        """
        allowed = list(POSTPROC_LOOKUP.keys())
        if instance_postproc not in allowed:
            raise ValueError(
                f"Illegal post proc method. Got {instance_postproc}. Allowed: {allowed}"
            )

        self.postproc_method = POSTPROC_LOOKUP[instance_postproc]
        self.use_argmax = INST_ARGMAX[instance_postproc]
        self.inst_key = inst_key
        self.aux_key = aux_key
        self.kwargs = kwargs

    def _get_sem_map(
        self,
        prob_map: np.ndarray,
        use_blur: bool = False,
        use_closing: bool = False,
        use_opening: bool = True,
        disk_size: int = 10,
        **kwargs,
    ) -> np.ndarray:
        """Run the semantic segmentation post-processing."""
        # Median filtering to get rid of noise. Adds a lot of overhead sop optional.
        if use_blur:
            sem = np.zeros_like(prob_map)
            for i in range(prob_map.shape[0]):
                sem[i, ...] = rank.median(
                    img_as_ubyte(prob_map[i, ...]), footprint=disk(disk_size)
                )
            prob_map = sem

        sem = np.argmax(prob_map, axis=0)

        if use_opening:
            sem = opening(sem, disk(disk_size))

        if use_closing:
            sem = closing(sem, disk(disk_size))

        sem = remove_debris_semantic(sem)
        sem = fill_holes_semantic(sem)

        return sem

    def _get_inst_map(
        self, prob_map: np.ndarray, aux_map: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Run the instance labelling post processing."""
        if self.use_argmax:
            inst = np.argmax(prob_map, axis=0)
        else:
            inst = prob_map.squeeze()

        return self.postproc_method(inst, aux_map, **self.kwargs)

    def _get_type_map(
        self,
        prob_map: np.ndarray,
        inst_map: np.ndarray,
        use_mask: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Run the type map post-processing. Majority voting for each instance.

        Adapted from:
        https://github.com/vqdang/hover_net/blob/master/models/hovernet/post_proc.py
        """
        type_map = np.argmax(prob_map, axis=0)
        if use_mask:
            type_map = binarize(inst_map) * type_map

        pred_id_list = np.unique(inst_map)[1:]
        for inst_id in pred_id_list:
            inst_type = type_map[inst_map == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            cell_type = type_list[0][0]

            if cell_type == 0:
                if len(type_list) > 1:
                    cell_type = type_list[1][0]

            type_map[inst_map == inst_id] = cell_type

        return type_map

    def post_proc_pipeline(
        self, maps: Dict[str, np.ndarray], **kwargs
    ) -> Dict[str, np.ndarray]:
        """Run post-processing for all the model outputs.

        Parameters
        ----------
            maps : Dict[str, np.ndarray]
                model head names mapped to model outputs.
                E.g. {"sem": np.ndarray, "type": np.ndarray, "cellpose": np.ndarray}

        Returns
        -------
            Dict[str, np.ndarray]:
                final output names mapped to the final post-processed outputs.
                E.g. {"sem": np.ndarray, "type": np.ndarray, "inst": np.ndarray}
        """
        res = {}
        if "sem" in maps.keys():
            res["sem"] = self._get_sem_map(maps["sem"], **self.kwargs)

        res["inst"] = self._get_inst_map(
            maps[self.inst_key], maps[self.aux_key], **self.kwargs
        )

        if "type" in maps.keys():
            res["type"] = self._get_type_map(maps["type"], res["inst"], **self.kwargs)

        return res

    def run_parallel(
        self, maps: List[Dict[str, np.ndarray]], progress_bar: bool = False
    ) -> List[Dict[str, np.ndarray]]:
        """Run the full post-processing pipeline in parallel for many model outputs.

        Parameters
        ----------
            maps : List[Dict[str, np.ndarray]]
                The model output map dictionaries in a list.
            progress_bar : bool, default=False
                If True, a tqdm progress bar is shown.


        Returns
        -------
            List[Dict[str, np.ndarray]]:
                The post-processed output map dictionaries in a list.
        """
        seg_results = []
        with Pool() as pool:
            if progress_bar:
                it = tqdm(pool.imap(self.post_proc_pipeline, maps), total=len(maps))
            else:
                it = pool.imap(self.post_proc_pipeline, maps)

            for x in it:
                seg_results.append(x)

        return seg_results
