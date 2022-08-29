from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base_inferer import BaseInferer

__all__ = ["SlidingWindowInferer"]


class SlidingWindowInferer(BaseInferer):
    def __init__(
        self,
        model: nn.Module,
        input_folder: Union[Path, str],
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        stride: int,
        patch_size: Tuple[int, int],
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
        """Sliding window inference for a folder of images.

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
            stride : int
                Stride of the sliding window.
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
                First n-number of images used from the `ìnput_folder`.
            **postproc_kwargs:
                Arbitrary keyword arguments for the post-processing.
        """
        super().__init__(
            model=model,
            input_folder=input_folder,
            out_activations=out_activations,
            out_boundary_weights=out_boundary_weights,
            patch_size=patch_size,
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

        self.stride = stride

    def _get_margins(
        self, first_endpoint: int, img_size: int, stride: int, pad: int = None
    ) -> Tuple[int, int]:
        """Get the number of slices needed for one direction and the overlap."""
        if pad is not None:
            img_size += int(pad)

        n = 1
        mod = 0
        end = first_endpoint
        while True:
            n += 1
            end += stride

            if end > img_size:
                mod = end - img_size
                break
            elif end == img_size:
                break

        return n, mod + pad

    def _get_slices(
        self,
        stride: int,
        patch_size: Tuple[int, int],
        img_size: Tuple[int, int],
        pad: int = None,
    ) -> Tuple[Dict[str, slice], int, int]:
        """Get all the overlapping slices in a dictionary and the needed paddings."""
        y_end, x_end = patch_size
        nrows, pady = self._get_margins(y_end, img_size[0], stride, pad=pad)
        ncols, padx = self._get_margins(x_end, img_size[1], stride, pad=pad)

        xyslices = {}
        for row in range(nrows):
            for col in range(ncols):
                y_start = row * stride
                y_end = y_start + patch_size[0]
                x_start = col * stride
                x_end = x_start + patch_size[1]
                xyslices[f"y-{y_start}_x-{x_start}"] = (
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                )

        return xyslices, pady, padx

    def _infer_batch(self, input_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Infer one batch of images."""
        slices, pady, padx = self._get_slices(
            self.stride, self.patch_size, tuple(input_batch.shape[2:]), self.padding
        )

        padx = padx // 2
        pady = pady // 2
        input_batch = F.pad(
            input_batch.float(), pad=(padx, padx, pady, pady), mode="reflect"
        )

        # initialize the output masks.
        out_maps = {}
        max_channels = 0
        for head_name, out_channels in self.out_heads:
            # use the largest out channel for recovery mask.
            max_channels = out_channels if out_channels > max_channels else max_channels

            out_maps[head_name] = torch.zeros(
                input_batch.shape[0],
                out_channels,
                *input_batch.shape[2:],
                dtype=input_batch.dtype,
                device=self.device,
            )

        # patches are added to the out mask so need a recovery mask.
        recovery = torch.zeros(
            input_batch.shape[0],
            max_channels,
            *input_batch.shape[2:],
            dtype=input_batch.dtype,
            device=self.device,
        )

        # run inference with the slices
        for k, (yslice, xslice) in slices.items():
            batch = input_batch[..., yslice, xslice].to(self.device).float()
            logits = self.predictor.forward_pass(batch)

            probs = {}
            for k, logit in logits.items():
                probs[k] = self.predictor.classify(logit, **self.head_kwargs[k])

            for k, out_map in out_maps.items():
                out_map[..., yslice, xslice] += probs[k]

            recovery[..., yslice, xslice] += 1

        for k, out_map in out_maps.items():
            out = out_map / recovery[:, 0 : out_map.shape[1], ...]
            out_maps[k] = out[..., pady:-pady, padx:-padx]

        return out_maps
