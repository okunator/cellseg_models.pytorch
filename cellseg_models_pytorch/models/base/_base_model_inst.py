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
        """Create a model from saved segmentation weights.

        The checkpoint determines the encoder name and number of cell-type
        output channels, plus the ray count for StarDist. Use the same
        preprocessing and class mapping as the training run. Call
        ``set_inference_mode`` before predicting images.

        Args:
            weights: Path to a local checkpoint, or a registered checkpoint name
                for this model family. Registered names download weights from
                Hugging Face when they are not already cached. Safetensors files
                and PyTorch state dictionaries must match this library's network.
            device: Device on which to load the weights and create the network.
                Pass ``torch.device("cpu")`` for CPU use; the default requires CUDA.
            model_kwargs: Additional network options needed to reproduce the
                checkpoint's architecture. Encoder name, class count and
                StarDist ray count are read from the weights.

        Returns:
            An instance of the model class used to call this method, with loaded
            weights. Inference is not yet enabled; its PyTorch network is
            available as ``model`` for training.

        Raises:
            ValueError: The path is not a file or a registered checkpoint name.
            ImportError: Loading a safetensors file requires safetensors.
            RuntimeError: The checkpoint weights do not match the network.

        Example:
            Load your checkpoint for image inference on the CPU:

            .. code-block:: python

                from pathlib import Path

                import torch

                from cellseg_models_pytorch.models.cellpose import CellPose

                model = CellPose.from_pretrained(
                    Path("checkpoints/cellpose.safetensors"),
                    device=torch.device("cpu"),
                )

            Then follow the quick start to configure prediction and reconstruction.
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

        enc_name, n_nuc_classes, state_dict = cls._get_state_dict(
            weights_path, device=device
        )

        inferred_kwargs: dict[str, int] = {}
        if cls.model_name == "stardist":
            ray_head_key = next(
                (
                    key
                    for key in state_dict
                    if key.endswith("nuc_stardist_head.head.weight")
                ),
                None,
            )
            if ray_head_key is None:
                raise RuntimeError(
                    "Checkpoint is missing the nucleus ray-head weights."
                )
            ray_shape = state_dict[ray_head_key].shape
            if len(ray_shape) != 4 or ray_shape[0] <= 0:
                raise RuntimeError(
                    "Checkpoint nucleus ray-head weights must have shape "
                    "(n_rays, channels, height, width) with n_rays > 0."
                )
            inferred_kwargs["n_rays"] = ray_shape[0]

        if weights_path.suffix == ".safetensors":
            # Release the first checkpoint copy before creating the model.
            del state_dict

        model_inst = cls(
            n_nuc_classes=n_nuc_classes,
            enc_name=enc_name,
            enc_pretrain=False,
            enc_freeze=False,
            device=device,
            model_kwargs=model_kwargs,
            **inferred_kwargs,
        )

        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_model
            except ImportError:
                raise ImportError(
                    "Please install `safetensors` package to load .safetensors files."
                )
            load_device = str(device) if device.type == "cuda" else device.type
            load_model(model_inst.model, weights_path, strict=True, device=load_device)
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
    ) -> dict[str, SoftSemanticOutput | SoftInstanceOutput | None]:
        """Predict class labels and reconstruction maps for an image or batch.

        Normalize inputs as required by the checkpoint. This method does not
        reconstruct individual objects; pass its result to ``post_process``.

        Args:
            x: RGB NumPy/PIL image with height, width and channels, or a tensor with
                channels first: (C, H, W) or (B, C, H, W). B is batch size, C channels,
                H height and W width. NumPy/PIL uint8 images are divided by 255;
                floating-point images keep their values. Tensor inputs must already
                have the network's device, floating dtype and expected value range.
            use_sliding_win: Predict overlapping windows and combine them into
                full-image maps. Each window must meet the network's size constraints.
            window_size: Window (height, width), required for sliding-window inference.
            stride: Window step in pixels, required with sliding windows. Must be
                positive and no larger than either window dimension.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` entries. Unused
            outputs are None. Instance entries contain ``type_map`` class labels
            of shape (B, H, W), or None without a cell-type head, and floating
            ``aux_map`` reconstruction values of
            shape (B, channels, H, W). These remain on the network device. A tissue
            entry contains class labels of shape (B, H, W). These are not probability
            maps. Class meanings and reconstruction maps depend on the checkpoint.

        Raises:
            ValueError: Inference has not been enabled, or the window settings or
                input dimensions are invalid.
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
        x: dict[str, SoftSemanticOutput | SoftInstanceOutput | None],
        *,
        use_async_postproc: bool = True,
        start_method: str = "threading",
        n_jobs: int = 4,
        save_paths_nuc: List[Union[Path, str]] = None,
        save_paths_cyto: List[Union[Path, str]] = None,
        coords: List[Tuple[int, int, int, int]] = None,
        class_dict_nuc: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
    ) -> dict[
        str, list[tuple[np.ndarray, np.ndarray | None] | np.ndarray | None] | None
    ]:
        """Turn predicted maps into individual nuclei and cell-type masks.

        Args:
            x: Dictionary returned by ``predict``, including a nucleus entry.
            use_async_postproc: Use asynchronous worker tasks for batches larger
                than one. This call waits for their results before returning.
                False uses synchronous workers; a single image always runs serially.
            start_method: Worker mode: ``threading``, ``fork`` or ``spawn``.
                Available process modes depend on the operating system.
            n_jobs: Number of reconstruction workers for batches larger than one.
            save_paths_nuc: One vector-output path per image for nuclei. Without
                paths, return masks in memory. Saving returns None for that image.
            save_paths_cyto: Equivalent paths for configured cytoplasm outputs.
            coords: One (x, y, width, height) tuple per image. Vector coordinates
                are offset by x and y when saving.
            class_dict_nuc: Map nucleus class numbers to names in saved vector data.
            class_dict_cyto: Map cytoplasm class numbers to names in saved vector data.

        Returns:
            Dictionary with ``nuc``, ``cyto`` and ``tissue`` entries. Nucleus and cytoplasm
            results are lists of (instances, classes) NumPy mask pairs, one per image.
            Both masks are int32 (H, W). Instance 0 is background; positive IDs
            identify objects. Each object gets the most common predicted class
            within its mask. Binary-only models return None for the class mask.
            Unused outputs and saved-image results are None.
            Tissue results are class masks. Tensor maps are copied to CPU.

        Raises:
            ValueError: Inference has not been enabled.

        Note:
            Use ``PostProcessor`` directly for tissue saving or custom contour
            smoothing. The selected model family controls the reconstruction method.
        """
        if not self.inference_mode:
            raise ValueError(
                "Run `.set_inference_mode()` before running `post_process`"
            )

        # if batch size is 1, run serially
        if x["nuc"].aux_map.shape[0] == 1:
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
    ) -> tuple[str | None, int, dict[str, torch.Tensor]]:
        """Load weights and return the encoder name, nucleus class count and tensors."""
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
            # Safetensors accepts CPU without an index; CUDA indices must be kept.
            load_device = str(device) if device.type == "cuda" else device.type
            state_dict = load_file(weights_path, device=load_device)
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
