from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm

from cellseg_models_pytorch.inference import BaseInferer

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

__all__ = ["INST_METRIC_LOOKUP", "SEM_METRIC_LOOKUP", "BenchMarker"]


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


class BenchMarker:
    def __init__(
        self,
        true_dir: str,
        pred_dir: str = None,
        inferer: BaseInferer = None,
        type_classes: Dict[str, int] = None,
        sem_classes: Dict[str, int] = None,
    ) -> None:
        """Run benchmarking, given prediction and ground truth mask folders.

        NOTE: Can also take in an Inferer object.

        Parameters
        ----------
            true_dir : str
                Path to the ground truth .mat files. The gt files have to have matching
                names to the pred filenames.
            pred_dir : str, optional
                Path to the prediction .mat files. The pred files have to have matching
                names to the gt filenames. If None, the inferer object storing the
                predictions will be used instead.
            inferer : BaseInferer, optional
                Infere object storing predictions of a model. If None, the `pred_dir`
                will be used to load the predictions instead.
            type_classes : Dict[str, int], optional
                Cell type class dict. E.g. {"bg": 0, "epithelial": 1, "immmune": 2}
            sem_classes : Dict[str, int], optional
                Tissue type class dict. E.g. {"bg": 0, "epithel": 1, "stroma": 2}
        """
        if pred_dir is None and inferer is None:
            raise ValueError(
                "Both `inferer` and `pred_dir` cannot be set to None at the same time."
            )

        self.true_dir = Path(true_dir)
        self.type_classes = type_classes
        self.sem_classes = sem_classes

        if pred_dir is not None:
            self.pred_dir = Path(pred_dir)
        else:
            self.pred_dir = None

        self.inferer = inferer

        if inferer is not None and pred_dir is None:
            try:
                self.inferer.out_masks
                self.inferer.soft_masks
            except AttributeError:
                raise AttributeError(
                    "Did not find `out_masks` or `soft_masks` attributes. "
                    "To get these, run inference with `inferer.infer()`. "
                    "Remember to set `save_intermediate` to True for the inferer.`"
                )

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

    def _compute_inst_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str]]]
    ) -> Dict[str, float]:
        return BenchMarker.compute_inst_metrics(*args)

    def _compute_sem_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str], int]]
    ) -> Dict[str, float]:
        return BenchMarker.compute_sem_metrics(*args)

    @staticmethod
    def run_metrics(
        args: List[Tuple[np.ndarray, np.ndarray, str, List[str]]],
        type: str = "inst",
        msg: str = None,
    ) -> List[Dict[str, float]]:
        """Run segmentation metrics in parallel.

        Parameters
        ----------
            args : List[Tuple[np.ndarray, np.ndarray, str, List[str]]]
                A list of params for `compute_inst_metrics` or `compute_sem_metrics`
            type : str, default="inst"
                One of "inst" or "sem".
            msg : str, optional
                msg for the progress bar.

        Returns
        -------
            List[Dict[str, float]]:
                A list of dictionaries where metric names are mapped to metric values.
        """
        f = (
            BenchMarker._compute_sem_metrics
            if type == "sem"
            else BenchMarker._compute_inst_metrics
        )

        info = "" if msg is None else msg
        metrics = []
        with Pool() as pool:
            for x in tqdm(
                pool.imap_unordered(f, args),
                total=len(args),
                desc=f"Running {info} metrics",
            ):
                metrics.append(x)

        return metrics

    def run_inst_benchmark(
        self, how: str = "binary", metrics: Tuple[str, ...] = ("pq",)
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

        trues = sorted(self.true_dir.glob("*"))

        preds = None
        if self.pred_dir is not None:
            preds = sorted(self.pred_dir.glob("*"))

        ik = "inst" if self.pred_dir is None else "inst_map"
        tk = "type" if self.pred_dir is None else "type_map"

        res = []
        if how == "multi" and self.type_classes is not None:
            for c, i in list(self.type_classes.items())[1:]:
                args = []
                for j, true_fn in enumerate(trues):
                    name = true_fn.name
                    true = FileHandler.read_mat(true_fn, return_all=True)

                    if preds is None:
                        pred = self.inferer.out_masks[name[:-4]]
                    else:
                        pred = FileHandler.read_mat(preds[j], return_all=True)

                    true_inst = true["inst_map"]
                    true_type = true["type_map"]
                    pred_inst = pred[ik]
                    pred_type = pred[tk]

                    pred_type = get_type_instances(pred_inst, pred_type, i)
                    true_type = get_type_instances(true_inst, true_type, i)
                    args.append((true_type, pred_type, name, metrics, c))
                met = self.run_metrics(args, "inst", f"{c} instance seg")
                res.extend([metric for metric in met if metric])
        else:
            args = []
            for i, true_fn in enumerate(trues):
                name = true_fn.name
                true = FileHandler.read_mat(true_fn, return_all=True)

                if preds is None:
                    pred = self.inferer.out_masks[name[:-4]]
                else:
                    pred = FileHandler.read_mat(preds[i], return_all=True)

                true = true["inst_map"]
                pred = pred[ik]
                args.append((true, pred, name, metrics))
            met = self.run_metrics(args, "inst", "binary instance seg")
            res.extend([metric for metric in met if metric])

        return res

    def run_sem_benchmark(self, metrics: Tuple[str, ...] = ("iou",)) -> Dict[str, Any]:
        """Run semantic segmentation benchmarking.

        Parameters
        ----------
            metrics : Tuple[str, ...], default=("iou", )
                A tuple of the metrics that will be computed. Allowed metrics:
                "iou", "acc", "f1", "dice" "sensitivity", "specificity".

        Returns
        -------
            Dict[str, Any]:
                Dictionary mapping the metrics to values + metadata.
        """
        trues = sorted(self.true_dir.glob("*"))

        preds = None
        if self.pred_dir is not None:
            preds = sorted(self.pred_dir.glob("*"))

        sk = "sem" if self.pred_dir is None else "sem_map"

        args = []
        for i, true_fn in enumerate(trues):
            name = true_fn.name
            true = FileHandler.read_mat(true_fn, return_all=True)

            if preds is None:
                pred = self.inferer.out_masks[name[:-4]]
            else:
                pred = FileHandler.read_mat(preds[i], return_all=True)
            true = true["sem_map"]
            pred = pred[sk]
            args.append((true, pred, name, len(self.sem_classes), metrics))

        met = self.run_metrics(args, "sem", "semantic seg")
        ires = [metric for metric in met if metric]

        # re-format
        res = []
        for r in ires:
            for k, val in self.sem_classes.items():
                cc = {
                    "name": r["name"],
                    "type": k,
                }
                for m in metrics:
                    cc[m] = r[m][val]
                res.append(cc)

        return res
