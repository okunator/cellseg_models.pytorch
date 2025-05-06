from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL.Image import Image

from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.models.base import PRETRAINED

__all__ = ["BaseModelInst"]


class BaseModelInst:
    def __init__(self) -> None:
        """Base class for cell segmentation models."""
        self.inference_mode = False

    def set_inference_mode(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        weights: Union[str, Path],
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> "BaseModelInst":
        """Load the model from pretrained weights.

        Parameters:
            model_name (str):
                Name of the pretrained model.
            device (torch.device, default=torch.device("cuda")):
                Device to run the model on. Default is "cuda".
            model_kwargs (Dict[str, Any], default={}):
                Additional arguments for the model.
        """
        weights_path = Path(weights)
        if not weights_path.is_file():
            if weights_path.as_posix() in PRETRAINED[cls.model_name].keys():
                weights_path = Path(
                    hf_hub_download(
                        repo_id=PRETRAINED[cls.model_name][weights]["repo_id"],
                        filename=PRETRAINED[cls.model_name][weights]["filename"],
                    )
                )

            else:
                raise ValueError(
                    "Please provide a valid path. or a pre-trained model downloaded from the"
                    f" csmp-hub. One of {list(PRETRAINED[cls.model_name].keys())}."
                )

        try:
            from safetensors.torch import load_model
        except ImportError:
            raise ImportError(
                "Please install `safetensors` package to load .safetensors files."
            )

        enc_name, n_nuc_classes, state_dict = cls._get_state_dict(
            weights_path, device=device
        )

        model_inst = cls(
            n_nuc_classes=n_nuc_classes,
            enc_name=enc_name,
            enc_pretrain=False,
            enc_freeze=False,
            device=device,
            model_kwargs=model_kwargs,
        )

        if weights_path.suffix == ".safetensors":
            load_model(model_inst.model, weights_path, device.type)
        else:
            model_inst.model.load_state_dict(state_dict, strict=True)

        return model_inst

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray, Image],
        *,
        use_sliding_win: bool = False,
        window_size: Tuple[int, int] = None,
        stride: int = None,
    ) -> Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]]:
        """Predict the input image or image batch.

        Parameters:
            x (Union[torch.Tensor, np.ndarray, Image]):
                Input image (H, W, C) or input image batch (B, C, H, W).
            use_sliding_win (bool, default=False):
                Whether to use sliding window for prediction.
            window_size (Tuple[int, int], default=None):
                The height and width of the sliding window. If `use_sliding_win` is False
                this argument is ignored.
            stride (int, default=None):
                The stride for the sliding window. If `use_sliding_win` is False this
                argument is ignored.

        Returns:
            Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]]:
                Dictionary of soft outputs:
                    - "nuc": SoftInstanceOutput (type_map, aux_map).
                    - "tissue": SoftSemanticOutput (type_map).
        """
        if not self.inference_mode:
            raise ValueError("Run `.set_inference_mode()` before running `predict`")

        if not use_sliding_win:
            x = self.predictor.predict(x=x, apply_boundary_weight=False)
        else:
            if window_size is None:
                raise ValueError(
                    "`window_size` must be provided when using sliding window."
                )
            if stride is None:
                raise ValueError("`stride` must be provided when using sliding window.")

            x = self.predictor.predict_sliding_win(
                x=x, window_size=window_size, stride=stride, apply_boundary_weight=True
            )

        return x

    def post_process(
        self,
        x: Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]],
        *,
        use_async_postproc: bool = True,
        start_method: str = "threading",
        n_jobs: int = 4,
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
    ) -> Dict[str, List[np.ndarray]]:
        """Post-process the output of the model.

        Parameters:
            x (Dict[str, Union[SoftSemanticOutput, SoftInstanceOutput]]):
                The output of the .predict() method.
            use_async_postproc (bool, default=True):
                Whether to use async post-processing. Can give some run-time benefits.
            start_method (str, default="threading"):
                The start method. One of: "threading", "fork", "spawn". See mpire docs.
            n_jobs (int, default=4):
                The number of workers for the post-processing.
            save_paths_nuc (List[Union[Path, str]], default=None):
                The paths to save the nuclei masks. If None, the masks are not saved.
            save_paths_cyto (List[Union[Path, str]], default=None):
                The paths to save the cytoplasm masks. If None, the masks are not saved.
            coords (List[Tuple[int, int, int, int]], default=None):
                The XYWH coordinates of the image patch. If not None, the coordinates are
                saved in the filenames of outputs.
            class_dict_nuc (Dict[int, str], default=None):
                The dictionary of nuclei classes. E.g. {0: "bg", 1: "neoplastic"}
            class_dict_cyto (Dict[int, str], default=None):
                The dictionary of cytoplasm classes. E.g. {0: "bg", 1: "macrophage_cyto"}

        Returns:
            Dict[str, List[np.ndarray]]:
                Dictionary of post-processed outputs:
                    - "nuc": List of output nuclei masks (H, W).
                    - "cyto": List of output cytoplasm masks (H, W).
        """
        if not self.inference_mode:
            raise ValueError(
                "Run `.set_inference_mode()` before running `post_process`"
            )

        # if batch size is 1, run serially
        if x["nuc"].type_map.shape[0] == 1:
            return self.post_processor.postproc_serial(
                x,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
            )

        if use_async_postproc:
            x = self.post_processor.postproc_parallel_async(
                x,
                start_method=start_method,
                n_jobs=n_jobs,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
            )
        else:
            x = self.post_processor.postproc_parallel(
                x,
                start_method=start_method,
                n_jobs=n_jobs,
                save_paths_nuc=save_paths_nuc,
                save_paths_cyto=save_paths_cyto,
                coords=coords,
                class_dict_nuc=class_dict_nuc,
                class_dict_cyto=class_dict_cyto,
            )

        return x

    @staticmethod
    def _get_state_dict(
        weights_path: Union[str, Path], device: torch.device = torch.device("cuda")
    ) -> None:
        """Load the model from pretrained weights."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ValueError(f"Model weights not found at {weights_path}")
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "Please install `safetensors` package to load .safetensors files."
                )
            state_dict = load_file(weights_path, device=device.type)
        else:
            state_dict = torch.load(weights_path, map_location=device)

        # infer encoder name and number of classes from state_dict
        enc_keys = [key for key in state_dict.keys() if "encoder." in key]
        enc_name = enc_keys[0].split(".")[0] if enc_keys else None
        nuc_type_head_key = next(
            key
            for key in state_dict.keys()
            if "nuc_type_head.head" in key and "weight" in key
        )
        n_nuc_classes = state_dict[nuc_type_head_key].shape[0]

        return enc_name, n_nuc_classes, state_dict
