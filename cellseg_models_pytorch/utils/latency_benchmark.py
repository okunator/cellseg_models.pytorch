from timeit import repeat
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from cellseg_models_pytorch.inference import BaseInferer

__all__ = ["LatencyBenchmarker"]


class LatencyBenchmarker:
    def __init__(self, inferer: BaseInferer) -> None:
        """Benchmark latencies of the model an post-processing pipelines.

        Parameters
        ----------
            inferer : BaseInferer
                An inferer object that contains model outputs.
        """
        self.inferer = inferer

        try:
            self.inferer.out_masks
            self.inferer.soft_masks
        except AttributeError:
            raise AttributeError(
                "Did not find `out_masks` or `soft_masks` attributes. "
                "To get these, run inference with `inferer.infer()`. "
                "Remember to set the `save_intermediate param to True for the inferer.`"
            )

    def inference_latency(
        self, reps: int = 1, warmup_reps: int = 1, **kwargs
    ) -> List[Dict[str, Any]]:
        """Compute the inference-pipeline latency.

        NOTE: computes only inference not post-processing latency.

        Parameters
        ----------
            reps : int, defalt=1
                Repetition per batch.
            warmup_reps : int, default=1
                Warm up repetitions.

        Returns
        -------
            List[Dict[str, Any]]:
                Latency info info of the samples.
        """
        times = []
        with tqdm(self.inferer.dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    res = self._compute_latency(
                        self.inferer._infer_batch,
                        maps={"input_batch": data["im"]},
                        reps=reps,
                        warmup_reps=warmup_reps,
                        **kwargs,
                    )
                    bsize = data["im"].shape[0]
                    res["n_images"] = bsize
                    res["input_shape"] = tuple(data["im"].shape[1:])
                    res["throughput (img/s)"] = res["mean latency(s)"] / bsize
                    times.append(res)

        return times

    def inference_postproc_latency(self, reps: int = 1) -> List[Dict[str, Any]]:
        """Compute the latency of the whole inference + post-processing pipeline.

        Parameters
        ----------
            reps : int, default=1
                Number of repetitions of the full pipeline.

        Returns
        -------
            List[Dict[str, Any]]
                The latency and throughput info of the pipeline.
        """
        timings = repeat(
            lambda: self.inferer.infer(),
            repeat=reps,
            number=1,
        )

        timings = np.array(timings)
        mean_syn = np.mean(timings)
        std_syn = np.std(timings)

        res = {}
        res["n_images"] = len(list(self.inferer.soft_masks.keys()))
        res["repetitions"] = reps
        res["total mean latency(s)"] = mean_syn
        res["total std latency(s)"] = std_syn
        res["mean throughput (img/s)"] = mean_syn / res["n_images"]
        res["std throughput (img/s))"] = std_syn / res["n_images"]

        return [res]

    def postproc_latency(
        self, which: str = "inst", reps_per_img: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """Compute the post-processing latencies.

        Parameters
        ----------
            which : str, default="inst"
                Which post-processing type. One of "inst", "sem", "type".
            reps_per_img : int, default=10
                Number of repetitions per image.
            **kwargs:
                Arbitrary keyword args for the post-proc func.

        Returns
        -------
            List[Dict[str, Any]]:
                A list of dicts containing throughput info of each of the samples.
        """
        PP_LOOKUP = {
            "sem": self._compute_sem_postproc_latency,
            "type": self._compute_type_postproc_latency,
            "inst": self._compute_inst_postproc_latency,
        }

        allowed = list(PP_LOOKUP.keys())
        if which not in allowed:
            raise ValueError(f"Illegal `type` arg. Got: {type}. Allowed: {allowed}")

        aux_key = self.inferer.postprocessor.aux_key
        inst_key = self.inferer.postprocessor.inst_key
        samples = list(self.inferer.soft_masks.keys())

        times = []
        with tqdm(samples, total=len(samples)) as pbar:
            for k in samples:
                if which == "inst":
                    res = PP_LOOKUP["inst"](
                        prob_map=self.inferer.soft_masks[k][inst_key],
                        aux_map=self.inferer.soft_masks[k][aux_key],
                        return_cell_count=True,
                        reps=reps_per_img,
                        **kwargs,
                    )
                elif which == "type":
                    res = PP_LOOKUP["type"](
                        prob_map=self.inferer.soft_masks[k][inst_key],
                        inst_map=self.inferer.out_masks[k]["inst"],
                        reps=reps_per_img,
                        **kwargs,
                    )
                elif which == "sem":
                    res = PP_LOOKUP["sem"](
                        prob_map=self.inferer.soft_masks[k]["sem"],
                        reps=reps_per_img,
                        **kwargs,
                    )

                res["name"] = k
                times.append(res)
                pbar.update(1)

        return times

    def model_latency(
        self,
        input_size: Tuple[int, int] = (256, 256),
        batch_size: int = 1,
        reps: int = 100,
        warmup_reps: int = 3,
        device="cuda",
    ) -> List[Dict[str, Any]]:
        """Measure the model inference latency in secods.

        I.e. one forward pass of the model.

        Parameters
        ----------
            input_size : Tuple[int, int]
                Height and width of the input.
            batch_size : int, default=1
                Batch size.
            reps : int, default=100
                Number of repetitions to run the latency measurement.
            warmup_reps : int, default=3
                Number of repetitions that are used for warming up the gpu.
                I.e. the number of repetitions that are excluded from the
                beginning.
            device : str, default="cuda"
                One of 'cuda' or 'cpu'.

        Returns
        -------
            List[Dict[str, Any]]:
                The latency mean and standard deviation in secods + extra info.
        """
        dummy_input = torch.randn(
            batch_size,
            3,
            input_size[0],
            input_size[1],
            dtype=torch.float,
            device=device,
        )

        if device == "cpu":
            self.inferer.predictor.model.cpu()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        timings = np.zeros((reps, 1))
        with tqdm(total=reps, unit="rep") as pbar:
            with torch.no_grad():
                for rep in range(reps):
                    starter.record()
                    _ = self.inferer.predictor.forward_pass(dummy_input)
                    ender.record()

                    # wait for gpu sync
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender) / 1000
                    timings[rep] = curr_time
                    pbar.update(1)

        mean_syn = np.sum(timings[warmup_reps:]) / (reps - warmup_reps)
        std_syn = np.std(timings[warmup_reps:])
        shape = tuple(dummy_input.shape[1:])

        res = {
            "batch_size": batch_size,
            "input_shape": shape,
            "mean latency(s)": mean_syn,
            "std latency(s)": std_syn,
        }

        return [res]

    def model_throughput(
        self,
        input_size: Tuple[int, int] = (256, 256),
        batch_size: int = 1,
        reps: int = 100,
        warmup_reps: int = 3,
        device="cuda",
    ) -> List[Dict[str, Any]]:
        """Measure the inference throughput in seconds.

        I.e. Measure model forward pass throughput (image/s).

        Parameters
        ----------
            input_size : Tuple[int, int]
                Height and width of the input.
            batch_size : int, default=1
                Batch size for the model.
            reps : int, default=300
                Number of repetitions to run the latency measurement.
            warmup_reps : int, default=3
                Number of repetitions that are used for warming up the gpu.
                I.e. the number of repetitions that are excluded from the
                beginning.
            device : str, default="cuda"
                One of 'cuda' or 'cpu'.

        Returns
        -------
            List[Dict[str, Any]]:
                The throughput of the model (image/s) + extra info.
        """
        dummy_input = torch.randn(
            batch_size,
            3,
            input_size[0],
            input_size[1],
            dtype=torch.float,
            device=device,
        )

        if device == "cpu":
            self.inferer.predictor.model.cpu()

        total_time = 0

        with torch.no_grad():
            for _ in range(reps):
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = self.inferer.predictor.forward_pass(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time

        throughput = ((reps - warmup_reps) * batch_size) / total_time
        shape = tuple(dummy_input.shape[1:])

        res = {
            "batch_size": batch_size,
            "input_shape": shape,
            "throughput(img/s)": throughput,
        }

        return [res]

    def _compute_latency(
        self,
        func: Callable,
        maps: Dict[str, Union[np.ndarray, torch.Tensor]],
        reps: int = 300,
        warmup_reps: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the latency measurements."""
        if kwargs:
            kwargs = {**maps, **kwargs}
        else:
            kwargs = maps

        timings = repeat(
            lambda: func(**kwargs),
            repeat=reps + warmup_reps,
            number=1,
        )
        timings = np.array(timings)

        mean_syn = np.mean(timings[warmup_reps:])
        std_syn = np.std(timings[warmup_reps:])

        res = {}
        res["repetitions"] = reps
        res["mean latency(s)"] = mean_syn
        res["std latency(s)"] = std_syn

        return res

    def _compute_inst_postproc_latency(
        self,
        prob_map: np.ndarray,
        aux_map: np.ndarray,
        reps: int = 10,
        warmup_reps: int = 2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute the instance segmentation post-proc latency in seconds.

        I.e. One run of the post-procesing method.

        NOTE: returns also additional data that affects the latency such
        as the number of objects and number of object pixels in the image.

        Parameters
        ----------
            prob_map : np.ndarray
                The probability map of the object instances. Shape: (C, H, W).
            aux_map : np.ndarray
                The auxilliary regression output. Shape: (C, H, W).
            reps : int, default=10
                Number of repetitions for timeit.
            warmup_reps : int, default=2
                Warmup loops for the function.
            **kwargs:
                Arbitrary keyword args for the post-proc func.

        Returns
        -------
            Dict[str, Any]:
                A dictionary with data related to the sample latency.
        """
        res = self._compute_latency(
            self.inferer.postprocessor._get_inst_map,
            maps={"prob_map": prob_map, "aux_map": aux_map},
            reps=reps,
            warmup_reps=warmup_reps,
            **kwargs,
        )

        x = self.inferer.postprocessor._get_inst_map(prob_map, aux_map, **kwargs)
        cells, counts = np.unique(x, return_counts=True)
        res["input_shape"] = x.shape
        res["ncells"] = len(cells)
        res["npixels"] = np.sum(counts[1:])

        return res

    def _compute_sem_postproc_latency(
        self, prob_map: np.ndarray, reps: int = 10, warmup_reps: int = 2, **kwargs
    ) -> List[Dict[str, Any]]:
        """Compute the semantic segmentation post-proc latency in seconds.

        I.e. One run of the post-procesing method.

        Parameters
        ----------
            prob_map : np.ndarray
                The probability map of the semantic segmentation. Shape: (C, H, W).
            reps : int, default=10
                Number of repetitions for timeit.
            warmup_reps : int, default=2
                Warmup loops for the function.
            **kwargs:
                Arbitrary keyword args for the post-proc func.

        Returns
        -------
            Dict[str, Any]:
                A dictionary with data related to the sample latency.
        """
        res = self._compute_latency(
            self.inferer.postprocessor._get_sem_map,
            maps={"prob_map": prob_map},
            reps=reps,
            warmup_reps=warmup_reps,
            **kwargs,
        )

        x = self.inferer.postprocessor._get_sem_map(prob_map, **kwargs)
        _, npixels = np.unique(x, return_counts=True)
        res["input_shape"] = x.shape
        res["npixels"] = np.sum(npixels[1:])

        return res

    def _compute_type_postproc_latency(
        self,
        prob_map: np.ndarray,
        inst_map: np.ndarray,
        reps: int = 10,
        warmup_reps: int = 2,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Compute the type segmentation post-proc latency in seconds.

        I.e. One run of the post-procesing method.

        Parameters
        ----------
            prob_map : np.ndarray
                The probability map of the object instances. Shape: (C, H, W).
            inst_map : np.ndarray
                The labelled instance segmentation map. Shape: (H, W).
            reps : int, default=10
                Number of repetitions for timeit.
            warmup_reps : int, default=2
                Warmup loops for the function.
            **kwargs:
                Arbitrary keyword args for the post-proc func.

        Returns
        -------
            Dict[str, Any]:
                A dictionary with data related to the sample latency.
        """
        res = self._compute_latency(
            self.inferer.postprocessor._get_type_map,
            maps={"prob_map": prob_map, "inst_map": inst_map},
            reps=reps,
            warmup_reps=warmup_reps,
            **kwargs,
        )

        x = self.inferer.postprocessor._get_type_map(
            prob_map, inst_map, use_mask=True, **kwargs
        )
        cells = np.unique(inst_map)
        _, npixels = np.unique(x, return_counts=True)
        res["img_shape"] = x.shape
        res["ncells"] = len(cells)
        res["npixels"] = np.sum(npixels[1:])

        return res
