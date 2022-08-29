from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base_inferer import BaseInferer


class ResizeInferer(BaseInferer):
    def __init__(
        self,
        model: nn.Module,
        input_folder: Union[Path, str],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        resize: Tuple[int, int],
        instance_postproc: str,
        padding: int = None,
        batch_size: int = 8,
        normalization: str = None,
        device: str = "cuda",
        save_masks: bool = True,
        save_intermediate: bool = False,
        save_dir: Union[Path, str] = None,
        checkpoint_path: Union[Path, str] = None,
        n_images: int = None,
        **postproc_kwargs,
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
            save_masks : bool, default=False
                If True, the resulting segmentation masks will be saved into `out_masks`
                variable.
            save_intermediate : bool, default=False
                If True, intermediate soft masks will be saved into `soft_masks` var.
            save_dir : bool, optional
                Path to save directory. If None, no masks will be saved to disk as .mat
                files. If not None, overrides `save_masks`, thus for every batch the
                segmentation results are saved into disk and the intermediate results
                are flushed.
            checkpoint_path : Path | str, optional
                Path to the model weight checkpoints.
            n_images : int, optional
                First n-number of images used from the `input_folder`.
            **postproc_kwargs:
                Arbitrary keyword arguments for the post-processing.
        """
        super().__init__(
            model=model,
            input_folder=input_folder,
            out_activations=out_activations,
            out_boundary_weights=out_boundary_weights,
            patch_size=resize,
            padding=padding,
            batch_size=batch_size,
            normalization=normalization,
            instance_postproc=instance_postproc,
            device=device,
            save_masks=save_masks,
            save_intermediate=save_intermediate,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path,
            n_images=n_images,
            **postproc_kwargs,
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
