from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base_inferer import BaseInferer


class ResizeInferer(BaseInferer):
    def __init__(
        self,
        model: nn.Module,
        input_path: Union[Path, str],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        resize: Tuple[int, int],
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
        """Resize inference for a folder of images.

        Resizes the image before inputting to the model.

        NOTE: This class assumes that all the images in the input folder
        have the same shape. If the images have different shapes,
        `batch_size` needs to be set to 1 for this to work.

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
            resize : Tuple[int, int]:
                The resized size for the input patches that are fed to the segmentation
                model.
            instance_postproc : str
                The post-processing method for the instance segmentation mask. One of:
                "cellpose", "omnipose", "stardist", "hovernet", "dcan", "drfns", "dran"
            padding : int, optional
                The amount of reflection padding for the input images.
            batch_size : int, default=8
                Number of images loaded from the folder at every batch.
            normalization : str, optional
                Apply img normalization (Same as during training). One of "dataset",
                "minmax", "norm", "percentile", None.
            device : str, default="cuda"
                The device of the input and model. One of: "cuda", "cpu"
            n_devices : int, default=1
                Number of devices (cpus/gpus) used for inference.
                The model will be copied into these devices.
            save_intermediate : bool, default=False
                If True, intermediate soft masks will be saved into `soft_masks` var.
            save_intermediate : bool, default=False
                If True, intermediate soft masks will be saved into `soft_masks` var.
            save_format : str, default=".mat"
                The file format for the saved output masks. One of (".mat", ".json").
                The ".json" option will save masks into geojson format.
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
                Arbitrary keyword arguments expecially for post-processing and saving.

        Examples
        --------
            >>> # initialize model and paths
            >>> model = cellpose_base(len(type_classes))
            >>> inputs = "/path/to/imgs"
            >>> ckpt_path = "/path/to/myweights.ckpt"

            >>> # initialize output head args
            >>> out_activations={"type": "softmax", "cellpose": None}
            >>> out_boundary_weights={"type": None, "cellpose": None}

            >>> inferer = ResizeInferer(
                    model=model,
                    input_path=inputs,
                    checkpoint_path=ckpt_path,
                    out_activations=out_activations,
                    out_boundary_weights=out_boundary_weights,
                    resize=(256, 256),
                    instance_postproc="cellpose",
                    padding=0,
                    normalization="minmax" # This needs to be same as during training
                )
            >>> inferer.infer()
        """
        super().__init__(
            model=model,
            input_path=input_path,
            out_activations=out_activations,
            out_boundary_weights=out_boundary_weights,
            patch_size=resize,
            padding=padding,
            batch_size=batch_size,
            normalization=normalization,
            instance_postproc=instance_postproc,
            device=device,
            n_devices=n_devices,
            save_intermediate=save_intermediate,
            save_dir=save_dir,
            save_format=save_format,
            checkpoint_path=checkpoint_path,
            n_images=n_images,
            type_post_proc=type_post_proc,
            sem_post_proc=sem_post_proc,
            **kwargs,
        )

    def _infer_batch(self, input_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Infer one batch of images."""
        inp_shape = tuple(input_batch.shape[2:])

        if self.padding:
            pad = self.padding // 2
            input_batch = F.pad(
                input_batch.float(), pad=(pad, pad, pad, pad), mode="reflect"
            )
            inp_shape = tuple(input_batch.shape[2:])

        input_batch = F.interpolate(input_batch, self.patch_size)
        batch = input_batch.to(self.device).float()
        logits = self.predictor.forward_pass(batch)

        probs = {}
        for k, logit in logits.items():
            prob = self.predictor.classify(logit, **self.head_kwargs[k])

            if self.padding:
                probs[k] = F.interpolate(prob, inp_shape)[..., pad:-pad, pad:-pad]
            else:
                probs[k] = F.interpolate(prob, inp_shape)

        return probs
