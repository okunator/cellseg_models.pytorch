from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import FileHandler, tensor_to_ndarray
from .folder_dataset_infer import FolderDatasetInfer
from .post_processor import PostProcessor
from .predictor import Predictor


class BaseInferer(ABC):
    def __init__(
        self,
        model: nn.Module,
        input_path: Union[Path, str],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        patch_size: Tuple[int, int],
        instance_postproc: str,
        padding: int = None,
        batch_size: int = 8,
        normalization: str = None,
        device: str = "cuda",
        n_devices: int = 1,
        checkpoint_path: Union[Path, str] = None,
        n_images: int = None,
        type_post_proc: Callable = None,
        sem_post_proc: Callable = None,
        **kwargs,
    ) -> None:
        """Inference for an image folder.

        Parameters
        ----------
        model : nn.Module
            A segmentation model.
        input_path : Path | str
            Path to a folder of images or to hdf5 db.
        out_activations : Dict[str, str]
            Dictionary of head names mapped to a string value that specifies the
            activation applied at the head. E.g. {"type": "tanh", "cellpose": None}
            Allowed values: "softmax", "sigmoid", "tanh", None.
        out_boundary_weights : Dict[str, bool]
            Dictionary of head names mapped to a boolean value. If the value is
            True, after a prediction, a weight matrix is applied that assigns bigger
            weight on pixels in the center and less weight to pixels on the tile
            boundaries. helps dealing with prediction artefacts on the boundaries.
            E.g. {"type": False, "cellpose": True}
        patch_size : Tuple[int, int]:
            The size of the input patches that are fed to the segmentation model.
        instance_postproc : str
            The post-processing method for the instance segmentation mask. One of:
            "cellpose", "omnipose", "stardist", "hovernet", "dcan", "drfns", "dran"
        padding : int, optional
            The amount of reflection padding for the input images.
        batch_size : int, default=8
            Number of images loaded from the folder at every batch.
        normalization : str, optional
            Apply img normalization at forward pass (Same as during training).
            One of: "dataset", "minmax", "norm", "percentile", None.
        device : str, default="cuda"
            The device of the input and model. One of: "cuda", "cpu"
        n_devices : int, default=1
            Number of devices (cpus/gpus) used for inference.
            The model will be copied into these devices.
        checkpoint_path : Path | str, optional
            Path to the model weight checkpoints.
        n_images : int, optional
            First n-number of images used from the `input_path`.
        type_post_proc : Callable, optional
            A post-processing function for the type maps. If not None, overrides
            the default.
        sem_post_proc : Callable, optional
            A post-processing function for the semantc seg maps. If not None,
            overrides the default.
        **kwargs:
            Arbitrary keyword arguments for post-processing.
        """
        # basic inits
        self.model = model
        self.out_heads = self._get_out_info()  # the names and num channels of out heads
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.padding = padding
        self.out_activations = out_activations
        self.out_boundary_weights = out_boundary_weights
        self.head_kwargs = self._check_and_set_head_args()
        self.kwargs = kwargs

        # dataset & dataloader
        self.path = Path(input_path)
        if self.path.is_dir():
            ds = FolderDatasetInfer(self.path, n_images=n_images)
        elif self.path.is_file() and self.path.suffix in (".h5", ".hdf5"):
            from .hdf5_dataset_infer import HDF5DatasetInfer

            ds = HDF5DatasetInfer(self.path, n_images=n_images)
        else:
            raise ValueError(
                f"Given `input_path`: {input_path} is neither an image folder or a h5 "
                "database. Allowed suffices for h5 database are ('.h5', '.hdf5')"
            )

        self.dataloader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        # Set post processor
        self.postprocessor = PostProcessor(
            instance_postproc,
            inst_key=self.model.inst_key,
            aux_key=self.model.aux_key,
            type_post_proc=type_post_proc,
            sem_post_proc=sem_post_proc,
            **kwargs,
        )

        # load weights and set devices
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            # check if path is url or local and load weigths to memory
            if urlparse(checkpoint_path.as_posix()).scheme:
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_path)
            else:
                state_dict = torch.load(
                    checkpoint_path, map_location=lambda storage, loc: storage
                )

            # if the checkpoint is from lightning, the ckpt file contains a lot of other
            # stuff than just the state dict.
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]

            # try loading the weights to the model
            try:
                msg = self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                new_ckpt = self._strip_state_dict(state_dict)
                msg = self.model.load_state_dict(new_ckpt, strict=True)
            except BaseException as e:
                raise RuntimeError(f"Error when loading checkpoint: {e}")

            print(f"Loading weights: {checkpoint_path} for inference.")
            print(msg)

        assert device in ("cuda", "cpu", "mps")
        if device == "cpu":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
            if torch.cuda.device_count() > 1 and n_devices > 1:
                self.model = nn.DataParallel(self.model, device_ids=range(n_devices))
        elif torch.backends.mps.is_available() and device == "mps":
            self.device = torch.device("mps")

        self.model.to(self.device)
        self.model.eval()

        # Helper class to perform forward + extra processing
        self.predictor = Predictor(
            model=self.model,
            patch_size=patch_size,
            normalization=normalization,
            device=self.device,
        )

    @classmethod
    def from_yaml(cls, model: nn.Module, yaml_path: str):
        """Initialize the inferer from a yaml-file.

        Parameters
        ----------
            model : nn.Module
                Initialized segmentation model.
            yaml_path : str
                Path to the yaml file containing rest of the params
        """
        with open(yaml_path, "r") as stream:
            kwargs = yaml.full_load(stream)

        return cls(model, **kwargs)

    @abstractmethod
    def _infer_batch(self):
        raise NotImplementedError

    def infer(
        self,
        save_dir: Union[Path, str] = None,
        save_format: str = ".mat",
        save_intermediate: bool = False,
        classes_type: Dict[str, int] = None,
        classes_sem: Dict[str, int] = None,
        offsets: bool = False,
        mixed_precision: bool = False,
    ) -> None:
        """Run inference and post-processing for the image(s) inside `input_path`.

        NOTE: If `save_dir` is None, the output masks will be cached in a class
        attribute `self.out_masks`. Otherwise the masks will be saved to disk.

        WARNING: Running inference without setting `save_dir` can take a lot of memory
        if the input directory contains many images.

        Parameters
        ----------
        save_dir : bool, optional
            Path to save directory. If None, no masks will be saved to disk.
            Instead the masks will be cached in a class attribute `self.out_masks`.
        save_format : str, default=".mat"
            The file format for the saved output masks. One of ".mat", ".geojson",
            "feather" "parquet".
        save_intermediate : bool, default=False
            If True, intermediate soft masks will be saved into `self.soft_masks`
            class attribute. WARNING: This can take a lot of memory if the input
            directory contains many images.
        classes_type : Dict[str, str], optional
            Cell type dictionary. e.g. {"inflam":1, "epithelial":2, "connec":3}.
            This is required only if `save_format` is one of the following formats:
            ".geojson", ".parquet", ".feather".
        classes_sem : Dict[str, str], otional
            Tissue type dictionary. e.g. {"tissue1":1, "tissue2":2, "tissue3":3}
            This is required only if `save_format` is one of the following formats:
            ".geojson", ".parquet", ".feather".
        offsets : bool, default=False
            If True, geojson coords are shifted by the offsets that are encoded in
            the filenames (e.g. "x-1000_y-4000.png"). Ignored if `format` == `.mat`.
        mixed_precision : bool, default=False
            If True, inference is performed with mixed precision.

        Attributes
        ----------
        - out_masks : Dict[str, Dict[str, np.ndarray]]
            The output masks for each image. The keys are the image names and the
            values are dictionaries of the masks. E.g.
            {"sample1": {"inst": [H, W], "type": [H, W], "sem": [H, W]}}
        - soft_masks : Dict[str, Dict[str, np.ndarray]]
            NOTE: This attribute is set only if `save_intermediate = True`.
            The soft masks for each image. I.e. the soft predictions of the trained
            model The keys are the image names and the values are dictionaries of
            the soft masks. E.g. {"sample1": {"type": [H, W], "aux": [C, H, W]}}
        """
        # check save_dir and save_format
        save_dir = Path(save_dir) if save_dir is not None else None
        save_intermediate = save_intermediate
        save_format = save_format
        if save_dir is not None:
            allowed_formats = (".mat", ".geojson", ".feather", ".parquet")
            if save_format not in allowed_formats:
                raise ValueError(
                    f"Given `save_format`: {save_format} is not one of the allowed "
                    f"formats: {allowed_formats}"
                )

        self.soft_masks = {}
        self.out_masks = {}
        self.elapsed = []
        self.rate = []
        with tqdm(self.dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    names = data["file"]
                    loader.set_description("Running inference")
                    loader.set_postfix_str("Forward pass")
                    soft_masks = self._infer_batch(data["im"], mixed_precision)
                    loader.set_postfix_str("post-processing")
                    soft_masks = self._prepare_mask_list(names, soft_masks)

                    # use multi-threading if batch size more than 1.
                    if self.batch_size > 1:
                        seg_results = self.postprocessor.run_parallel(soft_masks)
                    else:
                        seg_results = [
                            self.postprocessor.post_proc_pipeline(soft_masks[0])
                        ]

                    self.elapsed.append(loader.format_dict["elapsed"])
                    self.rate.append(loader.format_dict["rate"])

                    if save_intermediate:
                        for n, m in zip(names, soft_masks):
                            self.soft_masks[n] = m

                    # Quick kludge to add soft type and sem to seg_results
                    for soft, seg in zip(soft_masks, seg_results):
                        if "type" in soft.keys():
                            seg["soft_type"] = soft["type"]
                        if "sem" in soft.keys():
                            seg["soft_sem"] = soft["sem"]

                    # save to cache or disk
                    if save_dir is None:
                        for n, m in zip(names, seg_results):
                            self.out_masks[n] = m
                    else:
                        loader.set_postfix_str("Saving results to disk")
                        if self.batch_size > 1:
                            fnames = [Path(save_dir) / n for n in names]
                            FileHandler.save_masks_parallel(
                                maps=seg_results,
                                fnames=fnames,
                                format=save_format,
                                classes_type=classes_type,
                                classes_sem=classes_sem,
                                offsets=offsets,
                                pooltype="thread",
                                maptype="amap",
                            )
                        else:
                            for n, m in zip(names, seg_results):
                                fname = Path(save_dir) / n
                                FileHandler.save_masks(
                                    fname=fname,
                                    maps=m,
                                    format=save_format,
                                    classes_type=classes_type,
                                    classes_sem=classes_sem,
                                    offsets=offsets,
                                )

    def _strip_state_dict(self, ckpt: Dict) -> OrderedDict:
        """Strip te first 'model.' (generated by lightning) from the state dict keys."""
        state_dict = OrderedDict()
        for k, w in ckpt.items():
            if "num_batches_track" not in k:
                spl = ["".join(kk) for kk in k.split(".")]
                new_key = ".".join(spl[1:])
                state_dict[new_key] = w

        return state_dict

    def _check_and_set_head_args(self) -> None:
        """Check the model has matching head names with the head args and set them."""
        heads = [head_name for head_name, _ in self.out_heads]
        if self.out_boundary_weights.keys() != self.out_activations.keys():
            raise ValueError(
                "Mismatching head names in `out_boundary_weights` & `out_activations`",
            )

        pred_kwargs = {}
        for head_name, val in self.out_activations.items():
            if head_name in heads:
                pred_kwargs[head_name] = {"act": val}
                pred_kwargs[head_name]["apply_weights"] = self.out_boundary_weights[
                    head_name
                ]
            else:
                raise ValueError(
                    f"Mismatching head name. The model contains heads: {heads} ",
                    f"Got: {head_name}",
                )

        return pred_kwargs

    def _prepare_mask_list(
        self, fnames: List[str], soft_masks: torch.Tensor
    ) -> List[Dict[str, np.ndarray]]:
        """Convert the model outputs into arguments for parallel post-processing."""
        softies = {}
        for k, soft_mask in soft_masks.items():
            softies[k] = dict(zip(fnames, tensor_to_ndarray(soft_mask, "BCHW", "BCHW")))

        args = []
        for fn in fnames:
            map_dict = {}
            for mn, d in softies.items():
                map_dict[mn] = d[fn]
            args.append(map_dict)

        return args

    def _get_out_info(self) -> Tuple[Tuple[str, int]]:
        """Get the output names and number of out channels."""
        return tuple(
            chain.from_iterable(
                list(self.model.heads[k].items()) for k in self.model.heads.keys()
            )
        )
