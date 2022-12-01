from typing import Callable, Dict, List

import numpy as np
from skimage.util import img_as_ubyte

from ..postproc import POSTPROC_LOOKUP
from ..utils import (
    fill_holes_semantic,
    majority_vote_parallel,
    majority_vote_sequential,
    med_filt_parallel,
    med_filt_sequential,
    remove_debris_semantic,
    run_pool,
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


class PostProcessor:
    def __init__(
        self,
        instance_postproc: str,
        inst_key: str,
        aux_key: str,
        type_post_proc: Callable = None,
        sem_post_proc: Callable = None,
        **kwargs,
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
            type_post_proc : Callable, optional
                A post-processing function for the type maps. If not None, overrides
                the default.
            sem_post_proc : Callable, optional
                A post-processing function for the semantc seg maps. If not None,
                overrides the default.
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
        self.sem_post_proc = sem_post_proc
        self.type_post_proc = type_post_proc

    def _get_sem_map(
        self,
        prob_map: np.ndarray,
        parallel: bool = False,
        kernel_width: int = 15,
        **kwargs,
    ) -> np.ndarray:
        """Run the semantic segmentation post-processing."""
        sem_map = img_as_ubyte(prob_map)

        if self.sem_post_proc is not None:
            sem = self.sem_post_proc(sem_map)
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
        parallel: bool = True,
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
            res["inst"] *= res["type"] > 0

        return res

    def run_parallel(
        self,
        maps: List[Dict[str, np.ndarray]],
        pooltype: str = "thread",
        maptype: str = "amap",
    ) -> List[Dict[str, np.ndarray]]:
        """Run the full post-processing pipeline in parallel for many model outputs.

        Parameters
        ----------
            maps : List[Dict[str, np.ndarray]]
                The model output map dictionaries in a list.
            pooltype : str, default="thread"
                The pathos pooltype. Allowed: ("process", "thread", "serial").
                Defaults to "thread". (Fastest in benchmarks.)
            maptype : str, default="amap"
                The map type of the pathos Pool object.
                Allowed: ("map", "amap", "imap", "uimap")
                Defaults to "amap". (Fastest in benchmarks).

        Returns
        -------
            List[Dict[str, np.ndarray]]:
                The post-processed output map dictionaries in a list.
        """
        seg_results = run_pool(
            self.post_proc_pipeline, maps, ret=True, pooltype=pooltype, maptype=maptype
        )

        return seg_results
