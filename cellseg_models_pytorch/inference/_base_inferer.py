import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import FileHandler, tensor_to_ndarray
from .folder_dataset import FolderDataset
from .post_processor import PostProcessor
from .predictor import Predictor


class BaseInferer(ABC):
    def __init__(
        self,
        model: nn.Module,
        input_folder: Union[Path, str],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        patch_size: Tuple[int, int],
        instance_postproc: str,
        padding: int = None,
        batch_size: int = 8,
        normalization: str = None,
        device: str = "cuda",
        n_devices: int = 1,
        save_intermediate: bool = False,
        save_dir: Union[Path, str] = None,
        save_format: str = ".mat",
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
            input_folder : Path | str
                Path to a folder of images.
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
            save_dir : bool, optional
                Path to save directory. If None, no masks will be saved to disk as .mat
                or .json files. Instead the masks will be saved in `self.out_masks`.
            save_intermediate : bool, default=False
                If True, intermediate soft masks will be saved into `soft_masks` var.
            save_format : str, default=".mat"
                The file format for the saved output masks. One of (".mat", ".json").
                The ".json" option will save masks into geojson format.
            checkpoint_path : Path | str, optional
                Path to the model weight checkpoints.
            n_images : int, optional
                First n-number of images used from the `input_folder`.
            type_post_proc : Callable, optional
                A post-processing function for the type maps. If not None, overrides
                the default.
            sem_post_proc : Callable, optional
                A post-processing function for the semantc seg maps. If not None,
                overrides the default.
            **kwargs:
                Arbitrary keyword arguments expecially for post-processing and saving.
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

        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.save_intermediate = save_intermediate
        self.save_format = save_format

        # dataloader
        self.path = Path(input_folder)

        folder_ds = FolderDataset(self.path, n_images=n_images)
        if self.save_dir is None and len(folder_ds.fnames) > 40:
            warnings.warn(
                "`save_dir` is None. Thus, the outputs are be saved in `out_masks` "
                "class variable. If the input folder contains many images, running "
                "inference will likely flood the memory depending on the size and "
                "number of the images. Consider saving outputs to disk by providing "
                "`save_dir` argument."
            )

        self.dataloader = DataLoader(
            folder_ds, batch_size=batch_size, shuffle=False, pin_memory=True
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
            ckpt = torch.load(
                checkpoint_path, map_location=lambda storage, loc: storage
            )

            try:
                self.model.load_state_dict(ckpt["state_dict"], strict=True)
            except RuntimeError:
                new_ckpt = self._strip_state_dict(ckpt)
                self.model.load_state_dict(new_ckpt["state_dict"], strict=True)
            except BaseException as e:
                print(e)

        assert device in ("cuda", "cpu")
        if device == "cpu":
            self.device = torch.device("cpu")
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")

            if torch.cuda.device_count() > 1 and n_devices > 1:
                self.model = nn.DataParallel(self.model, device_ids=range(n_devices))

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

    def infer(self) -> None:
        """Run inference and post-processing for the images.

        NOTE:
        - Saves outputs in `self.out_masks` or to disk (.mat/.json) files.
        - If `save_intermediate` is set to True, also intermiediate model outputs are
            saved to `self.soft_masks`
        - `self.out_masks` and `self.soft_masks` are nested dicts: E.g.
                {"sample1": {"inst": [H, W], "type": [H, W], "sem": [H, W]}}
        - If masks are saved to geojson .json files, more key word arguments
            need to be given at class initialization. Namely: `geo_format`,
            `classes_type`, `classes_sem`, `offsets`. See more in the
            `FileHandler.save_masks` docs.
        """
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
                    soft_masks = self._infer_batch(data["im"])
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

                    if self.save_intermediate:
                        for n, m in zip(names, soft_masks):
                            self.soft_masks[n] = m

                    if self.save_dir is None:
                        for n, m in zip(names, seg_results):
                            self.out_masks[n] = m
                    else:
                        loader.set_postfix_str("Saving results to disk")
                        if self.batch_size > 1:
                            fnames = [Path(self.save_dir) / n for n in names]
                            FileHandler.save_masks_parallel(
                                maps=seg_results,
                                fnames=fnames,
                                **{**self.kwargs, "format": self.save_format},
                            )
                        else:
                            for n, m in zip(names, seg_results):
                                fname = Path(self.save_dir) / n
                                FileHandler.save_masks(
                                    fname=fname,
                                    maps=m,
                                    **{**self.kwargs, "format": self.save_format},
                                )

    def _strip_state_dict(self, ckpt: Dict) -> OrderedDict:
        """Strip te first 'model.' (generated by lightning) from the state dict keys."""
        state_dict = OrderedDict()
        for k, w in ckpt["state_dict"].items():
            if "num_batches_track" not in k:
                new_key = k.strip("model")[1:]
                state_dict[new_key] = w
        ckpt["state_dict"] = state_dict

        return ckpt

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
