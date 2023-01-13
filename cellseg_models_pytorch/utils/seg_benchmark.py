import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from ..inference import BaseInferer
from ..metrics import (
    accuracy_multiclass,
    aggregated_jaccard_index,
    average_precision,
    dice2,
    dice_multiclass,
    f1score_multiclass,
    iou_multiclass,
    panoptic_quality,
    sensitivity_multiclass,
    specificity_multiclass,
)
from .file_manager import FileHandler
from .mask_utils import get_type_instances, remap_label
from .multiproc import run_pool

__all__ = ["SegBenchmarker"]


INST_METRIC_LOOKUP = {
    "pq": panoptic_quality,
    "aji": aggregated_jaccard_index,
    "dice2": dice2,
    "ap": average_precision,
}


SEM_METRIC_LOOKUP = {
    "iou": iou_multiclass,
    "dice": dice_multiclass,
    "f1": f1score_multiclass,
    "acc": accuracy_multiclass,
    "sensitivity": sensitivity_multiclass,
    "specificity": specificity_multiclass,
}


class SegBenchmarker:
    def __init__(
        self,
        true_path: str,
        pred_path: str = None,
        inferer: BaseInferer = None,
        type_classes: Dict[str, int] = None,
        sem_classes: Dict[str, int] = None,
    ) -> None:
        """Run benchmarking, given prediction and ground truth mask folders or hdf5 dbs.

        NOTE: Can also take in an Inferer object that contains predictions.

        Parameters
        ----------
            true_path : str
                Path to the ground truth .mat files or h5 db. The gt files have to have
                matching names to the pred filenames.
            pred_path : str, optional
                Path to thse prediction .mat files. The pred files have to have matching
                names to the gt filenames. If None, the inferer object storing the
                predictions will be used instead.
            inferer : BaseInferer, optional
                Infere object storing predictions of a model. If None, the `pred_path`
                will be used to load the predictions instead.
            type_classes : Dict[str, int], optional
                Cell type class dict. E.g. {"bg": 0, "epithelial": 1, "immmune": 2}
            sem_classes : Dict[str, int], optional
                Tissue type class dict. E.g. {"bg": 0, "epithel": 1, "stroma": 2}

        Example
        -------
            >>> # define type model, classes and paths
            >>> type_classes = {"bg": 0, "infl": 1, "neoplastic": 2}
            >>> model = cellpose_base(len(type_classes))
            >>> im_dir = "/path/to/ims"
            >>> lab_dir = "/path/to/labels/"

            >>> # Run inference first
            >>> inferer = ResizeInferer(
                    model=model,
                    input_path=im_dir,
                    checkpoint_path=ckpt_path,
                    out_activations={"type": "softmax", "cellpose": None},
                    out_boundary_weights={"type": None, "cellpose": None},
                    resize=(256, 256),
                    instance_postproc="cellpose",
                    normalization="minmax"
                )
            >>> inferer.infer()

            >>> bm = SegBenchmarker(
                    true_path=lab_dir,
                    inferer=inferer,
                    type_classes=type_classes
                )

            >>> # Run multi-class instance segmentation benchmarking
            >>> bm.run_inst_benchmark(how="multi")
        """
        if pred_path is None and inferer is None:
            raise ValueError(
                "Both `inferer` and `pred_path` cannot be set to None at the same time."
            )
        self.type_classes = type_classes
        self.sem_classes = sem_classes

        # solve the input paths, whether folder or h5 db.
        self.true_path = Path(true_path)
        if self.true_path.is_dir():
            self.true_ftype = "folder"
        elif self.true_path.is_file() and self.true_path.suffix in (".h5", ".hdf5"):
            self.true_ftype = "h5"
        else:
            raise ValueError(
                "The `true_path` needs to be a folder of .mat files or a h5 db."
            )

        if pred_path is not None:
            self.pred_path = Path(pred_path)
            if self.pred_path.is_dir():
                self.pred_ftype = "folder"
            elif self.pred_path.is_file() and self.pred_path.suffix in (".h5", ".hdf5"):
                self.pred_ftype = "h5"
            else:
                raise ValueError(
                    "The `true_path` needs to be a folder of .mat files or a h5 db."
                )
        else:
            self.pred_path = None
            self.pred_ftype = "inferer"
            try:
                inferer.out_masks
                inferer.soft_masks
                self.inferer = inferer
            except AttributeError:
                raise AttributeError(
                    "Did not find `out_masks` or `soft_masks` attributes. "
                    "To get these, run inference with `inferer.infer()`. "
                    "Remember to set `save_intermediate` to True for the inferer.`"
                )

        # define whether input files are from folder or h5-db
        if self.true_ftype == "folder":
            self.true_files = sorted(self.true_path.glob("*"))
            self.n_items = len(self.true_files)
        else:
            self.true_files = self.true_path
            self.n_items = FileHandler.read_h5_patch(
                self.true_files, 0, False, False, False, return_nitems=True
            )["nitems"]

        if self.pred_ftype == "folder":
            self.pred_files = sorted(self.pred_path.glob("*"))
        elif self.pred_ftype == "h5":
            self.pred_files = self.pred_path
        else:
            self.pred_files = None

    @staticmethod
    def compute_inst_metrics(
        true: np.ndarray,
        pred: np.ndarray,
        name: str,
        metrics: Tuple[str, ...] = ("pq",),
        type: str = "binary",
    ) -> Dict[str, float]:
        """Compute metrics for one prediciton-ground truth pair.

        Parameters
        ----------
            true : np.ndarray
                The ground truth instance labelled mask. Shape: (H, W).
            pred : np.ndarray
                The predicted instance labelled mask. Shape: (H, W).
            name : str
                Name of the sample/image.
            metrics : Tuple[str, ...], default=("pq", )
                A tuple of the metrics that will be computed. Allowed metrics:
                "pq", "aji", "dice2", "split_and_merge".

        Raises
        ------
            ValueError if an illegal metric is given.

        Returns
        -------
            Dict[str, float]:
                A dictionary where metric names are mapped to metric values.
                e.g. {"pq": 0.5, "aji": 0.55, "name": "sample1"}
        """
        allowed = list(INST_METRIC_LOOKUP.keys())
        if not all([m in allowed for m in metrics]):
            raise ValueError(
                f"An illegal metric was given. Got: {metrics}, allowed: {allowed}"
            )

        # Do not run metrics computation if there are no instances in neither of masks
        res = {}
        if len(np.unique(true)) > 1 or len(np.unique(pred)) > 1:
            true = remap_label(true)
            pred = remap_label(pred)

            met = {}
            for m in metrics:
                met[m] = INST_METRIC_LOOKUP[m]

            for k, m in met.items():
                score = m(true, pred)

                if isinstance(score, dict):
                    for n, v in score.items():
                        res[n] = v
                else:
                    res[k] = score

            res["name"] = name
            res["type"] = type
        else:
            res["name"] = name
            res["type"] = type

            for m in metrics:
                if m == "pq":
                    res["pq"] = -1.0
                    res["sq"] = -1.0
                    res["dq"] = -1.0
                else:
                    res[m] = -1.0

        return res

    @staticmethod
    def compute_sem_metrics(
        true: np.ndarray,
        pred: np.ndarray,
        name: str,
        num_classes: int,
        metrics: Tuple[str, ...] = ("iou",),
    ) -> Dict[str, float]:
        """Compute metrics for one prediciton-ground truth pair.

        Parameters
        ----------
            true : np.ndarray
                The ground truth semantic seg mask. Shape: (H, W).
            pred : np.ndarray
                The predicted semantic seg mask. Shape: (H, W).
            name : str
                Name of the sample/image.
            num_classes : int
                Number of classes in the dataset.
            metrics : Tuple[str, ...], default=("iou", )
                A tuple of the metrics that will be computed. Allowed metrics:
                "iou", "aji", "dice2", "split_and_merge".

        Raises
        ------
            ValueError if an illegal metric is given.

        Returns
        -------
            Dict[str, float]:
                A dictionary where metric names are mapped to metric values.
                e.g. {"iou": 0.5, "f1score": 0.55, "name": "sample1"}
        """
        if not isinstance(metrics, tuple) and not isinstance(metrics, list):
            raise ValueError("`metrics` must be either a list or tuple of values.")

        allowed = list(SEM_METRIC_LOOKUP.keys())
        if not all([m in allowed for m in metrics]):
            raise ValueError(
                f"An illegal metric was given. Got: {metrics}, allowed: {allowed}"
            )

        # Skip empty GTs
        met = {}
        for m in metrics:
            met[m] = SEM_METRIC_LOOKUP[m]

        res = {}
        for k, m in met.items():
            score = m(true, pred, num_classes=num_classes)
            res[k] = score

        res["name"] = name
        return res

    @staticmethod
    def run_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str]]],
        type: str = "inst",
    ) -> List[Dict[str, float]]:
        """Run segmentation metrics in parallel.

        Parameters
        ----------
            args : List[Tuple[np.ndarray, np.ndarray, str, List[str]]]
                A list of params for `compute_inst_metrics` or `compute_sem_metrics`
            type : str, default="inst"
                One of "inst" or "sem".

        Returns
        -------
            List[Dict[str, float]]:
                A list of dictionaries where metric names are mapped to metric values.
        """
        f = (
            SegBenchmarker._compute_sem_metrics
            if type == "sem"
            else SegBenchmarker._compute_inst_metrics
        )

        metrics = run_pool(f, args)

        return metrics

    def run_inst_benchmark(
        self, how: str, metrics: Tuple[str, ...] = ("pq",), chunk_size: int = 10
    ) -> None:
        """Run instance segmentation benchmarking.

        Parameters
        ----------
            how : str, default="binary"
                One of "binary" or "multi". "multi" computes metrics per class and
                "binary" computes metrics for all objects.
            metrics : Tuple[str, ...], default=("pq", )
                A tuple of the metrics that will be computed. Allowed metrics:
                "pq", "aji", "dice2" "ap".
            chunk_size : int, default=10
                Number of images in one chunk. Each chunk is send to multiprocessing.
                The benchmarking is by default run in chunks to not overflood the mem.

        Raises
        ------
            ValueError if an illegal `how` arg is given.

        Returns
        -------
            Dict[str, Any]:
                Dictionary mapping the metrics to values + metadata.
        """
        allowed = ("binary", "multi")
        if how not in allowed:
            raise ValueError(f"Illegal arg `how`. Got: {how}, Allowed: {allowed}")

        res = []
        if how == "multi" and self.type_classes is not None:
            for c, i in list(self.type_classes.items())[1:]:
                msg = c
                res.extend(
                    self._send_chunks_to_multiprocess(
                        metrics, "inst", msg, how=how, cls_val=i, chunk_size=chunk_size
                    )
                )
        else:
            msg = "binary"
            res.extend(
                self._send_chunks_to_multiprocess(
                    metrics, "inst", msg, how=how, chunk_size=chunk_size
                )
            )

        return res

    def run_sem_benchmark(
        self, metrics: Tuple[str, ...] = ("iou",), chunk_size: int = 10
    ) -> None:
        """Run semantic segmentation benchmarking.

        Parameters
        ----------
            metrics : Tuple[str, ...], default=("pq", )
                A tuple of the metrics that will be computed. Allowed metrics:
                "pq", "aji", "dice2" "ap".
            chunk_size : int, default=10
                Number of images in one chunk. Each chunk is send to multiprocessing.
                The benchmarking is by default run in chunks to not overflood the mem.


        Returns
        -------
            Dict[str, Any]:
                Dictionary mapping the metrics to values + metadata.
        """
        res = []
        msg = "semantic"
        res.extend(
            self._send_chunks_to_multiprocess(
                metrics, "sem", msg, chunk_size=chunk_size
            )
        )

        # re-format
        out = []
        for r in res:
            for k, val in self.sem_classes.items():
                datum = {"name": r["name"], "type": k}
                for m in metrics:
                    datum[m] = r[m][val]
                out.append(datum)

        return out

    def _read_one_mask_set(
        self,
        path: Union[Path, List[Path]],
        ix: int,
        ftype: str,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        name: str = None,
    ) -> Dict[str, np.ndarray]:
        """Read a set of masks from a .mat file, h5-db or inferer object."""
        out = {}

        if ftype == "folder":
            name = path[ix].with_suffix("").name
            out["name"] = name

            masks = FileHandler.read_mat(path[ix], return_all=True)
            if return_inst:
                out["inst"] = masks["inst_map"]
            if return_type:
                out["type"] = masks["type_map"]
            if return_sem:
                out["sem"] = masks["sem_map"]
        elif ftype == "h5":
            masks = FileHandler.read_h5_patch(
                path=path,
                ix=ix,
                return_im=False,
                return_inst=return_inst,
                return_type=return_type,
                return_sem=return_sem,
                return_name=True,
            )
            out["name"] = masks["name"].with_suffix("").name

            if return_inst:
                out["inst"] = masks["inst"]
            if return_type:
                out["type"] = masks["type"]
            if return_sem:
                out["sem"] = masks["sem"]
        else:
            if name is None:
                name = path[ix].with_suffix("").name

            out["name"] = name
            masks = self.inferer.out_masks[name]
            if return_inst:
                out["inst"] = masks["inst"]
            if return_type:
                out["type"] = masks["type"]
            if return_sem:
                out["sem"] = masks["sem"]

        return out

    def _read_masks(
        self,
        true_path: Union[Path, List[Path]],
        pred_path: Union[Path, List[Path]],
        ix: int,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Read a set of pred & gt masks from either .mat files, h5 db or an inferer."""
        true_masks = self._read_one_mask_set(
            true_path,
            ix,
            self.true_ftype,
            return_inst=return_inst,
            return_type=return_type,
            return_sem=return_sem,
        )

        pred_masks = self._read_one_mask_set(
            pred_path if pred_path is not None else true_path,
            ix,
            self.pred_ftype,
            return_inst=return_inst,
            return_type=return_type,
            return_sem=return_sem,
            name=true_masks["name"] if self.pred_ftype == "inferer" else None,
        )

        return true_masks, pred_masks

    def _send_chunks_to_multiprocess(
        self,
        metrics: Tuple[str, ...],
        type: str,
        msg: str,
        how: str = None,
        chunk_size: int = 10,
        **kwargs,
    ) -> List[Dict[str, np.ndarray]]:
        """Send chunks one by one to multiprocessing."""
        res = []

        pbar_msg = f"Running {msg} {type} benchmark: "
        with tqdm(
            list(self._chunks(range(self.n_items), chunk_size)), desc=pbar_msg
        ) as pbar:
            for chunk in pbar:
                if type == "inst":
                    args = self._set_inst_args(
                        chunk, how=how, metrics=metrics, cls_name=msg, **kwargs
                    )
                else:
                    args = self._set_sem_args(chunk, metrics)

                met = self.run_metrics(args, type)
                res.extend([metric for metric in met if metric])

        return res

    def _set_inst_args(
        self,
        chunk: Iterable,
        how: str,
        metrics: Tuple[str, ...],
        cls_name: str = None,
        cls_val: int = None,
        **kwargs,
    ) -> List[Tuple[Any]]:
        """Generate arguments for instance seg benchmarking in parallel."""
        args = []
        for i in chunk:
            true_masks, pred_masks = self._read_masks(
                true_path=self.true_files,
                pred_path=self.pred_files,
                ix=i,
                return_inst=True,
                return_type=how == "multi",
            )

            # set args for inst benchmark
            name = true_masks["name"]
            if how == "multi":
                true_inst = true_masks["inst"]
                true_type = true_masks["type"]
                pred_inst = pred_masks["inst"]
                pred_type = pred_masks["type"]
                pred_type = get_type_instances(pred_inst, pred_type, cls_val)
                true_type = get_type_instances(true_inst, true_type, cls_val)
                arg_set = (true_type, pred_type, name, metrics, cls_name)
            else:
                true_inst = true_masks["inst"]
                pred_inst = pred_masks["inst"]
                arg_set = (true_inst, pred_inst, name, metrics)

            args.append(arg_set)

        return args

    def _set_sem_args(
        self,
        chunk: Iterable,
        metrics: Tuple[str, ...],
    ) -> List[Tuple[Any]]:
        """Generate arguments for semantic seg benchmarking in parallel."""
        args = []
        for i in chunk:
            true_masks, pred_masks = self._read_masks(
                true_path=self.true_files,
                pred_path=self.pred_files,
                ix=i,
                return_inst=False,
                return_type=False,
                return_sem=True,
            )
            # set args for inst benchmark
            name = true_masks["name"]
            args.append(
                (
                    true_masks["sem"],
                    pred_masks["sem"],
                    name,
                    len(self.sem_classes),
                    metrics,
                )
            )

        return args

    def _chunks(self, iterable: Iterable, size: int) -> Iterable:
        """Generate adjacent chunks of an iterable."""
        it = iter(iterable)
        return iter(lambda: tuple(itertools.islice(it, size)), ())

    def _compute_inst_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str]]]
    ) -> Dict[str, float]:
        return SegBenchmarker.compute_inst_metrics(*args)

    def _compute_sem_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str], int]]
    ) -> Dict[str, float]:
        return SegBenchmarker.compute_sem_metrics(*args)
