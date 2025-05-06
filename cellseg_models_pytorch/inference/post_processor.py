import gc
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from mpire import WorkerPool

from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.postproc import POSTPROC_LOOKUP
from cellseg_models_pytorch.utils import (
    FileHandler,
    fill_holes_semantic,
    majority_vote_sequential,
    remove_debris_semantic,
)
from cellseg_models_pytorch.utils.vectorize import gaussian_smooth, inst2gdf, sem2gdf

__all__ = ["PostProcessor"]


class PostProcessor:
    def __init__(self, postproc_method: str, postproc_kwargs: dict = {}) -> None:
        """A class for post-processing of all the different types of model outputs.

        Parameters:
            postproc_method (str):
                The post-processing method for the instance segmentation mask.
            postproc_kwargs (dict):
                Arbitrary post-processing kwargs for the postproc method.

        Raises:
            ValueError: If the post-processing method is not allowed.
        """
        allowed = list(POSTPROC_LOOKUP.keys())
        if postproc_method not in allowed:
            raise ValueError(
                f"Illegal post proc method. Got {postproc_method}. Allowed: {allowed}"
            )

        self.postproc_func = partial(
            POSTPROC_LOOKUP[postproc_method], **postproc_kwargs
        )

    def postproc_tissuemap(
        self,
        tissue_map: np.ndarray,
        save_path: Union[Path, str] = None,
        coords: Tuple[int, int, int, int] = None,
        class_dict: Dict[str, int] = None,
    ) -> np.ndarray:
        """Run tissue map post-processing."""
        tissue_map = remove_debris_semantic(tissue_map, min_size=5000)
        tissue_map = fill_holes_semantic(tissue_map, min_size=5000).astype("i4")

        if save_path is not None:
            self._save_sem2vector(save_path, tissue_map, coords, class_dict)
            gc.collect()
        else:
            gc.collect()
            return tissue_map

    def postproc_inst(
        self,
        inst_map: np.ndarray,
        aux_map: np.ndarray,
        type_map: np.ndarray,
        save_path: Union[Path, str] = None,
        coords: Tuple[int, int, int, int] = None,
        class_dict: Dict[str, int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run instace map post-processing."""
        inst_map = self.postproc_func(inst_map, aux_map).astype("i4")
        type_map = majority_vote_sequential(type_map, inst_map).astype("i4")

        if save_path is not None:
            self._save_inst2vector(save_path, inst_map, type_map, coords, class_dict)
            gc.collect()
        else:
            gc.collect()
            return inst_map, type_map

    def postproc_parallel(
        self,
        soft_masks,
        n_jobs: int = 4,
        start_method: str = "threading",
        progress_bar: bool = False,
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        save_paths_tissue: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Post-process the masks in parallel using multiprocessing."""
        # set up input args for
        nuc_inst_maps, nuc_aux_maps, nuc_type_maps = self._prepare_inst_maps(
            soft_masks["nuc"]
        )
        if save_paths_nuc is None:
            save_paths_nuc = [None] * len(nuc_inst_maps)

        if soft_masks["cyto"] is not None:
            cyto_inst_maps, cyto_aux_maps, cyto_type_maps = self._prepare_inst_maps(
                soft_masks["cyto"]
            )
            if save_paths_cyto is None:
                save_paths_cyto = [None] * len(cyto_inst_maps)

        if soft_masks["tissue"] is not None:
            tissue_maps = self._prepare_tissue_maps(soft_masks["tissue"])
            if save_paths_tissue is None:
                save_paths_tissue = [None] * len(tissue_maps)

        if coords is None:
            coords = [None] * len(nuc_inst_maps)

        nuc_results = None
        cyto_results = None
        tissue_results = None

        # set the number of jobs to 1 if there is only one instance map
        if len(nuc_inst_maps) == 1:
            n_jobs = 1

        with WorkerPool(
            n_jobs=n_jobs, start_method=start_method, keep_alive=True, use_dill=True
        ) as pool:
            if soft_masks["nuc"] is not None:
                nuc_results = self._pool_map(
                    pool,
                    partial(self.postproc_inst, class_dict=class_dict_nuc),
                    list(
                        zip(
                            nuc_inst_maps,
                            nuc_aux_maps,
                            nuc_type_maps,
                            save_paths_nuc,
                            coords,
                        )
                    ),
                    progress_bar=progress_bar,
                )

            if soft_masks["cyto"] is not None:
                cyto_results = self._pool_map(
                    pool,
                    partial(self.postproc_inst, class_dict=class_dict_cyto),
                    list(
                        zip(
                            cyto_inst_maps,
                            cyto_aux_maps,
                            cyto_type_maps,
                            save_paths_cyto,
                            coords,
                        )
                    ),
                    progress_bar=progress_bar,
                )

            if soft_masks["tissue"] is not None:
                tissue_results = self._pool_map(
                    pool,
                    partial(self.postproc_tissuemap, class_dict=class_dict_tissue),
                    list(zip(tissue_maps, save_paths_tissue, coords)),
                    progress_bar=progress_bar,
                )

        return {"nuc": nuc_results, "cyto": cyto_results, "tissue": tissue_results}

    def postproc_parallel_async(
        self,
        soft_masks,
        n_jobs: int = 4,
        start_method: str = "threading",
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        save_paths_tissue: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Post-process the masks in parallel using async."""
        # set up input args for
        nuc_inst_maps, nuc_aux_maps, nuc_type_maps = self._prepare_inst_maps(
            soft_masks["nuc"]
        )
        if save_paths_nuc is None:
            save_paths_nuc = [None] * len(nuc_inst_maps)

        if soft_masks["cyto"] is not None:
            cyto_inst_maps, cyto_aux_maps, cyto_type_maps = self._prepare_inst_maps(
                soft_masks["cyto"]
            )
            if save_paths_cyto is None:
                save_paths_cyto = [None] * len(cyto_inst_maps)

        if soft_masks["tissue"] is not None:
            tissue_maps = self._prepare_tissue_maps(soft_masks["tissue"])
            if save_paths_tissue is None:
                save_paths_tissue = [None] * len(tissue_maps)

        if coords is None:
            coords = [None] * len(nuc_inst_maps)

        # run post-processing
        nuc_results = None
        cyto_results = None
        tissue_results = None

        # set the number of jobs to 1 if there is only one instance map
        if len(nuc_inst_maps) == 1:
            n_jobs = 1

        with WorkerPool(
            n_jobs=n_jobs, start_method=start_method, keep_alive=True, use_dill=True
        ) as pool:
            if soft_masks["nuc"] is not None:
                nuc_results = self._pool_apply_async(
                    pool,
                    partial(self.postproc_inst, class_dict=class_dict_nuc),
                    list(
                        zip(
                            nuc_inst_maps,
                            nuc_aux_maps,
                            nuc_type_maps,
                            save_paths_nuc,
                            coords,
                        )
                    ),
                )

            if soft_masks["cyto"] is not None:
                cyto_results = self._pool_apply_async(
                    pool,
                    partial(self.postproc_inst, class_dict=class_dict_cyto),
                    list(
                        zip(
                            cyto_inst_maps,
                            cyto_aux_maps,
                            cyto_type_maps,
                            save_paths_cyto,
                            coords,
                        )
                    ),
                )

            if soft_masks["tissue"] is not None:
                tissue_results = self._pool_apply_async(
                    pool,
                    partial(self.postproc_tissuemap, class_dict=class_dict_tissue),
                    list(zip(tissue_maps, save_paths_tissue, coords)),
                )

            pool.stop_and_join()

        nuc_results = [async_result.get() for async_result in nuc_results]
        tissue_results = [async_result.get() for async_result in tissue_results]

        return {"nuc": nuc_results, "cyto": cyto_results, "tissue": tissue_results}

    def postproc_serial(
        self,
        soft_masks,
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        save_paths_tissue: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Run post-processing sequentially."""
        nuc_inst_maps, nuc_aux_maps, nuc_type_maps = self._prepare_inst_maps(
            soft_masks["nuc"]
        )
        if save_paths_nuc is None:
            save_paths_nuc = [None] * len(nuc_inst_maps)

        if soft_masks["cyto"] is not None:
            cyto_inst_maps, cyto_aux_maps, cyto_type_maps = self._prepare_inst_maps(
                soft_masks["cyto"]
            )
            if save_paths_cyto is None:
                save_paths_cyto = [None] * len(cyto_inst_maps)

        if soft_masks["tissue"] is not None:
            tissue_maps = self._prepare_tissue_maps(soft_masks["tissue"])
            if save_paths_tissue is None:
                save_paths_tissue = [None] * len(tissue_maps)

        if coords is None:
            coords = [None] * len(nuc_inst_maps)

        nuc_results = None
        cyto_results = None
        tissue_results = None

        if soft_masks["nuc"] is not None:
            nuc_results = [
                self.postproc_inst(
                    inst_map, aux_map, type_map, save_path, coord, class_dict_nuc
                )
                for inst_map, aux_map, type_map, save_path, coord in zip(
                    nuc_inst_maps,
                    nuc_aux_maps,
                    nuc_type_maps,
                    save_paths_nuc,
                    coords,
                )
            ]

        if soft_masks["cyto"] is not None:
            cyto_results = [
                self.postproc_inst(
                    inst_map, aux_map, type_map, save_path, coord, class_dict_cyto
                )
                for inst_map, aux_map, type_map, save_path, coord in zip(
                    cyto_inst_maps,
                    cyto_aux_maps,
                    cyto_type_maps,
                    save_paths_cyto,
                    coords,
                )
            ]

        if soft_masks["tissue"] is not None:
            tissue_results = [
                self.postproc_tissuemap(tissue_map, save_path, coord, class_dict_tissue)
                for tissue_map, save_path, coord in zip(
                    tissue_maps, save_paths_tissue, coords
                )
            ]

        return {"nuc": nuc_results, "cyto": cyto_results, "tissue": tissue_results}

    def _pool_apply_async(self, pool: WorkerPool, func, args):
        async_result = [pool.apply_async(func, args=arg) for arg in args]
        return async_result

    def _pool_map(self, pool: WorkerPool, func, args, progress_bar: bool = False):
        map_results = pool.map(
            func,
            args,
            progress_bar=progress_bar,
            concatenate_numpy_output=False,
        )
        return map_results

    def _to_ndarray(self, tensor: torch.Tensor, dtype: str) -> np.ndarray:
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.is_cuda:
            tensor = tensor.cpu()

        return tensor.numpy().astype(dtype)

    def _prepare_inst_maps(
        self, soft_masks: SoftInstanceOutput
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if soft_masks.binary_map is not None:
            prob_maps = soft_masks.binary_map
        else:
            prob_maps = soft_masks.type_map
        prob_maps = prob_maps.squeeze(1)

        inst_dtype = "i4" if soft_masks.binary_map is None else "f4"
        inst_maps = list(self._to_ndarray(prob_maps, dtype=inst_dtype))
        aux_maps = list(self._to_ndarray(soft_masks.aux_map, dtype="f4"))
        type_maps = list(self._to_ndarray(soft_masks.type_map, dtype="i4"))

        return inst_maps, aux_maps, type_maps

    def _prepare_tissue_maps(self, soft_masks_tissue: SoftSemanticOutput):
        if soft_masks_tissue.binary_map is not None:
            prob_maps = soft_masks_tissue.binary_map
        else:
            prob_maps = soft_masks_tissue.type_map

        tissue_maps = list(self._to_ndarray(prob_maps, dtype="i4"))

        return tissue_maps

    def _save_inst2vector(
        self,
        save_path: Union[Path, str],
        inst_map: np.ndarray,
        type_map: np.ndarray,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict: dict = None,
        compute_centroids: bool = False,
        compute_bboxes: bool = False,
    ) -> None:
        save_path = Path(save_path)

        xoff = coords[0] if coords is not None else None
        yoff = coords[1] if coords is not None else None

        inst_gdf = inst2gdf(
            inst_map,
            type_map,
            xoff=xoff,
            yoff=yoff,
            class_dict=class_dict,
            smooth_func=gaussian_smooth,
        )

        if compute_centroids:
            inst_gdf["centroid"] = inst_gdf["geometry"].centroid
        if compute_bboxes:
            inst_gdf["bbox"] = inst_gdf["geometry"].apply(lambda x: x.bounds)

        FileHandler.gdf_to_file(inst_gdf, save_path, silence_warnings=True)

    def _save_sem2vector(
        self,
        save_path: Union[Path, str],
        sem_map: np.ndarray,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict: dict = None,
    ) -> None:
        save_path = Path(save_path)

        xoff = coords[0] if coords is not None else None
        yoff = coords[1] if coords is not None else None

        sem_gdf = sem2gdf(
            sem_map,
            xoff=xoff,
            yoff=yoff,
            class_dict=class_dict,
        )

        FileHandler.gdf_to_file(sem_gdf, save_path, silence_warnings=True)
