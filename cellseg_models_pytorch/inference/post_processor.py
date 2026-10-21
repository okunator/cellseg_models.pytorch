import gc
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

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
    """Reconstruct individual objects and assign a class to each object.

    Choose the method that matches your model family.
    Batch methods copy tensor maps to CPU NumPy arrays to separate objects.
    """

    def __init__(self, postproc_method: str, postproc_kwargs: dict = {}) -> None:
        """Choose the reconstruction method and its options.

        Args:
            postproc_method: Method name, for example ``cellpose``, ``omnipose``,
                ``hovernet``, ``stardist`` or ``instanseg``. Must match the model output.
            postproc_kwargs: Options passed to that method. For CellPose, use
                ``{"use_gpu": False}`` for CPU reconstruction. InstanSeg needs the
                trained network's ``pixel_classifier``.

        Raises:
            ValueError: The method name is unknown.
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
        class_dict: dict[int, str] | None = None,
        smooth_func: Callable = None,
    ) -> np.ndarray | None:
        """Clean one tissue-class mask and optionally save polygons.

        Args:
            tissue_map: Integer class mask (H, W). Small regions below 5000 pixels
                are removed and holes are filled before output.
            save_path: Optional vector output path; without it, return the mask.
            coords: Optional (x, y, width, height) tuple. Saved geometry is offset
                by x and y; masks returned in memory keep local pixel coordinates.
            class_dict: Map integer class numbers to names for vector output.
            smooth_func: Optional polygon-smoothing function used only when saving.

        Returns:
            Cleaned int32 class mask (H, W), or None when saving polygons.
        """
        tissue_map = remove_debris_semantic(tissue_map, min_size=5000)
        tissue_map = fill_holes_semantic(tissue_map, min_size=5000).astype("i4")

        if save_path is not None:
            self._save_sem2vector(
                save_path, tissue_map, coords, class_dict, smooth_func=smooth_func
            )
            gc.collect()
        else:
            gc.collect()
            return tissue_map

    def postproc_inst(
        self,
        inst_map: np.ndarray,
        aux_map: np.ndarray,
        type_map: np.ndarray | None = None,
        save_path: Path | str | None = None,
        coords: tuple[int, int, int, int] | None = None,
        class_dict: dict[int, str] | None = None,
        smooth_func: Callable | None = gaussian_smooth,
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Reconstruct one image's objects and vote on their classes.

        Args:
            inst_map: HW foreground labels/scores expected by the reconstruction method.
                This is the foreground input, not already reconstructed instance IDs.
            aux_map: Floating family-specific reconstruction map (channels, H, W).
            type_map: Integer class labels (H, W) for majority voting, or None
                when the model has no cell-type head.
            save_path: Optional vector file path (.parquet, .geojson or .feather).
                Existing-file behavior follows the selected writer.
            coords: Optional (x, y, width, height) tuple. Saved geometry is offset
                by x and y; masks returned in memory keep local pixel coordinates.
            class_dict: Map integer class numbers to names for vector output.
            smooth_func: Optional polygon-smoothing function used only when saving.

        Returns:
            (instances, classes) as int32 NumPy arrays of shape (H, W). Instance 0 is
            background; positive IDs identify objects. Returns None instead when
            saving to a file. Smoothing affects saved polygons, not returned masks.
            The classes entry is None for binary-only models; saved polygons then
            use the vector writer's generic foreground class 1.
        """
        inst_map = self.postproc_func(inst_map, aux_map).astype("i4")
        if type_map is not None:
            type_map = majority_vote_sequential(type_map, inst_map).astype("i4")

        if save_path is not None:
            self._save_inst2vector(
                save_path,
                inst_map,
                type_map,
                coords,
                class_dict,
                smooth_func=smooth_func,
            )
            gc.collect()
        else:
            gc.collect()
            return inst_map, type_map

    def postproc_parallel(
        self,
        soft_masks: dict[str, SoftInstanceOutput | SoftSemanticOutput | None],
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
        nuc_smooth_func: Callable = gaussian_smooth,
        cyto_smooth_func: Callable = gaussian_smooth,
        tissue_smooth_func: Callable = None,
    ) -> dict[
        str, list[tuple[np.ndarray, np.ndarray | None] | np.ndarray | None] | None
    ]:
        """Reconstruct a batch using workers.

        Args:
            soft_masks: Prediction dictionary with a nucleus entry and optional
                ``cyto``/``tissue`` entries. Class maps must already contain class
                labels, as returned by ``Predictor``, not raw class logits.
            n_jobs: Number of workers; a batch of one uses one worker.
            start_method: Worker mode: "threading", "fork" or "spawn", subject to
                operating-system support.
            progress_bar: Show worker progress.
            save_paths_nuc: Optional vector output paths, one per image, for nuclei.
            save_paths_cyto: Equivalent paths for cytoplasm.
            save_paths_tissue: Equivalent paths for tissue.
            coords: Optional (x, y, width, height) per image for vector coordinate offsets.
            class_dict_nuc: Map nucleus class numbers to output names.
            class_dict_cyto: Map cytoplasm class numbers to output names.
            class_dict_tissue: Map tissue class numbers to output names.
            nuc_smooth_func: Polygon smoothing for saved nuclei; None disables it.
            cyto_smooth_func: Polygon smoothing for saved cytoplasm; None disables it.
            tissue_smooth_func: Polygon smoothing for saved tissue; None disables it.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` lists, in input order.
            Missing heads are None. Instance results are (instances, classes) int32
            NumPy masks (H, W); classes are None without a cell-type head.
            Tissue results are int32 class masks (H, W).
            Saving an image returns None at that list position. All worker results
            are resolved before returning; no future is returned to the caller.
        """
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
                    partial(
                        self.postproc_inst,
                        class_dict=class_dict_nuc,
                        smooth_func=nuc_smooth_func,
                    ),
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
                    partial(
                        self.postproc_inst,
                        class_dict=class_dict_cyto,
                        smooth_func=cyto_smooth_func,
                    ),
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
                    partial(
                        self.postproc_tissuemap,
                        class_dict=class_dict_tissue,
                        smooth_func=tissue_smooth_func,
                    ),
                    list(zip(tissue_maps, save_paths_tissue, coords)),
                    progress_bar=progress_bar,
                )

        return {"nuc": nuc_results, "cyto": cyto_results, "tissue": tissue_results}

    def postproc_parallel_async(
        self,
        soft_masks: dict[str, SoftInstanceOutput | SoftSemanticOutput | None],
        n_jobs: int = 4,
        start_method: str = "threading",
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        save_paths_tissue: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
        nuc_smooth_func: Callable = gaussian_smooth,
        cyto_smooth_func: Callable = gaussian_smooth,
        tissue_smooth_func: Callable = None,
    ) -> dict[
        str, list[tuple[np.ndarray, np.ndarray | None] | np.ndarray | None] | None
    ]:
        """Schedule batch reconstruction on workers and wait for all results.

        Args:
            soft_masks: Prediction dictionary with a nucleus entry and optional
                ``cyto``/``tissue`` entries. Class maps must already contain class
                labels, as returned by ``Predictor``, not raw class logits.
            n_jobs: Number of workers; a batch of one uses one worker.
            start_method: Worker mode: "threading", "fork" or "spawn", subject to
                operating-system support.
            save_paths_nuc: Optional vector output paths, one per image, for nuclei.
            save_paths_cyto: Equivalent paths for cytoplasm.
            save_paths_tissue: Equivalent paths for tissue.
            coords: Optional (x, y, width, height) per image for vector coordinate offsets.
            class_dict_nuc: Map nucleus class numbers to output names.
            class_dict_cyto: Map cytoplasm class numbers to output names.
            class_dict_tissue: Map tissue class numbers to output names.
            nuc_smooth_func: Polygon smoothing for saved nuclei; None disables it.
            cyto_smooth_func: Polygon smoothing for saved cytoplasm; None disables it.
            tissue_smooth_func: Polygon smoothing for saved tissue; None disables it.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` lists, in input order.
            Missing heads are None. Instance results are (instances, classes) int32
            NumPy masks (H, W); classes are None without a cell-type head.
            Tissue results are int32 class masks (H, W).
            Saving an image returns None at that list position. All worker results
            are resolved before returning; no future is returned to the caller.
        """
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
                    partial(
                        self.postproc_inst,
                        class_dict=class_dict_nuc,
                        smooth_func=nuc_smooth_func,
                    ),
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
                    partial(
                        self.postproc_inst,
                        class_dict=class_dict_cyto,
                        smooth_func=cyto_smooth_func,
                    ),
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
                    partial(
                        self.postproc_tissuemap,
                        class_dict=class_dict_tissue,
                        smooth_func=tissue_smooth_func,
                    ),
                    list(zip(tissue_maps, save_paths_tissue, coords)),
                )

            pool.stop_and_join()

        if nuc_results is not None:
            nuc_results = [async_result.get() for async_result in nuc_results]
        if cyto_results is not None:
            cyto_results = [async_result.get() for async_result in cyto_results]
        if tissue_results is not None:
            tissue_results = [async_result.get() for async_result in tissue_results]

        return {"nuc": nuc_results, "cyto": cyto_results, "tissue": tissue_results}

    def postproc_serial(
        self,
        soft_masks: dict[str, SoftInstanceOutput | SoftSemanticOutput | None],
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        save_paths_tissue: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        class_dict_tissue: Dict[int, str] = None,
        nuc_smooth_func: Callable = gaussian_smooth,
        cyto_smooth_func: Callable = gaussian_smooth,
        tissue_smooth_func: Callable = None,
    ) -> dict[
        str, list[tuple[np.ndarray, np.ndarray | None] | np.ndarray | None] | None
    ]:
        """Reconstruct a batch one image at a time.

        Args:
            soft_masks: Prediction dictionary with a nucleus entry and optional
                ``cyto``/``tissue`` entries. Class maps must already contain class
                labels, as returned by ``Predictor``, not raw class logits.
            save_paths_nuc: Optional vector output paths, one per image, for nuclei.
            save_paths_cyto: Equivalent paths for cytoplasm.
            save_paths_tissue: Equivalent paths for tissue.
            coords: Optional (x, y, width, height) per image for vector coordinate offsets.
            class_dict_nuc: Map nucleus class numbers to output names.
            class_dict_cyto: Map cytoplasm class numbers to output names.
            class_dict_tissue: Map tissue class numbers to output names.
            nuc_smooth_func: Polygon smoothing for saved nuclei; None disables it.
            cyto_smooth_func: Polygon smoothing for saved cytoplasm; None disables it.
            tissue_smooth_func: Polygon smoothing for saved tissue; None disables it.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` lists, in input order.
            Missing heads are None. Instance results are (instances, classes) int32
            NumPy masks (H, W); classes are None without a cell-type head.
            Tissue results are int32 class masks (H, W).
            Saving an image returns None at that list position. All worker results
            are resolved before returning; no future is returned to the caller.
        """
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
                    inst_map,
                    aux_map,
                    type_map,
                    save_path,
                    coord,
                    class_dict_nuc,
                    smooth_func=nuc_smooth_func,
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
                    inst_map,
                    aux_map,
                    type_map,
                    save_path,
                    coord,
                    class_dict_cyto,
                    smooth_func=cyto_smooth_func,
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
                self.postproc_tissuemap(
                    tissue_map,
                    save_path,
                    coord,
                    class_dict_tissue,
                    smooth_func=tissue_smooth_func,
                )
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

        if tensor.dtype == torch.bfloat16:
            # NumPy cannot store bfloat16; float32 preserves its values exactly.
            return tensor.float().numpy().astype(dtype, copy=False)

        return tensor.numpy().astype(dtype)

    def _prepare_inst_maps(
        self, soft_masks: SoftInstanceOutput
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
        """Copy a batch's foreground, reconstruction and optional class maps to CPU."""
        if soft_masks.binary_map is not None:
            prob_maps = soft_masks.binary_map
        else:
            prob_maps = soft_masks.type_map
        if prob_maps is None:
            raise ValueError("Instance reconstruction requires a binary or class map.")
        prob_maps = prob_maps.squeeze(1)

        inst_dtype = "i4" if soft_masks.binary_map is None else "f4"
        inst_maps = list(self._to_ndarray(prob_maps, dtype=inst_dtype))
        aux_maps = list(self._to_ndarray(soft_masks.aux_map, dtype="f4"))
        # Missing class predictions need no full-size placeholder mask.
        type_maps = (
            list(self._to_ndarray(soft_masks.type_map, dtype="i4"))
            if soft_masks.type_map is not None
            else [None] * len(inst_maps)
        )

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
        save_path: Path | str,
        inst_map: np.ndarray,
        type_map: np.ndarray | None,
        coords: tuple[int, int, int, int] | None = None,
        class_dict: dict[int, str] | None = None,
        compute_centroids: bool = False,
        compute_bboxes: bool = False,
        smooth_func: Callable | None = gaussian_smooth,
    ) -> None:
        """Save one image's instances and optional cell types as polygons."""
        save_path = Path(save_path)

        xoff = coords[0] if coords is not None else None
        yoff = coords[1] if coords is not None else None

        inst_gdf = inst2gdf(
            inst_map,
            type_map,
            xoff=xoff,
            yoff=yoff,
            class_dict=class_dict,
            smooth_func=smooth_func,
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
        smooth_func: Callable = None,
    ) -> None:
        save_path = Path(save_path)

        xoff = coords[0] if coords is not None else None
        yoff = coords[1] if coords is not None else None

        sem_gdf = sem2gdf(
            sem_map,
            xoff=xoff,
            yoff=yoff,
            class_dict=class_dict,
            smooth_func=smooth_func,
        )

        FileHandler.gdf_to_file(sem_gdf, save_path, silence_warnings=True)
