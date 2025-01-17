from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .inferer import Inferer

__all__ = ["SlidingWindowInferer"]


class SlidingWindowInferer(Inferer):
    def __init__(
        self,
        model: torch.nn.Module,
        patch_shape: Tuple[int, int],
        stride: int,
        out_activations: Dict[str, str],
        out_boundary_weights: Dict[str, bool],
        post_proc_method: str,
        padding: int = 20,
        num_post_proc_threads: int = -1,
        mixed_precision: bool = False,
        **post_proc_kwargs,
    ) -> None:
        """The SlidingWindowInferer class runs Inference using a sliding window approach.
        The benefit of this approach is that it can handle large images that do not fit
        in GPU memory. However, it is often slower than the regular `Inferer`

        Parameters:
            model (torch.nn.Module):
                The neural network model to be used for inference.
            patch_shape (Tuple[int, int]):
                The shape of the sliding window patches (H, W).
            stride (int):
                The stride of the sliding window.
            out_activations (Dict[str, str]):
                Dictionary specifying the activation functions for the output heads.
            out_boundary_weights (Dict[str, bool]):
                Dictionary specifying whether boundary weights should be used for each
                output head.
            post_proc_method (str):
                The method to be used for post-processing.
            padding (int, default=20):
                Padding to be added to the original input to do sliding window slicing.
            num_post_proc_threads (int, default=-1):
                Number of threads to be used for post-processing.
                If -1, all available CPUs are used.
            mixed_precision (bool, default=False):
                Whether to use mixed precision during inference.
            **post_proc_kwargs:
                Additional keyword arguments for the PostProcessor instance.

        Raises:
            ValueError: If an invalid activation is specified in out_activations.

        Attributes:
            model (torch.nn.Module): The neural network model.
            pool (ThreadPool): Thread pool for post-processing.
            weight_mat (torch.Tensor, optional): Boundary weight matrix.
            post_processor (PostProcessor): Post-processor instance.

        Methods:
            predict(x: torch.Tensor, output_shape: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
            Run model prediction on the input tensor and return the model predictions.

            post_process(probs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
            Post-process the model predictions and return the post-processed predictions.

            post_process_parallel(probs: Dict[str, torch.Tensor], maptype: str = "amap") -> List[Dict[str, np.ndarray]]:
            Run the full post-processing pipeline in parallel for many model outputs and return the post-processed outputs.
        """
        super().__init__(
            model=model,
            input_shape=patch_shape,
            out_activations=out_activations,
            out_boundary_weights=out_boundary_weights,
            post_proc_method=post_proc_method,
            num_post_proc_threads=num_post_proc_threads,
            mixed_precision=mixed_precision,
            **post_proc_kwargs,
        )

        self.patch_shape = patch_shape
        self.stride = stride
        self.padding = padding

    def _get_margins(
        self, first_endpoint: int, img_size: int, stride: int, pad: int = None
    ) -> Tuple[int, int]:
        """Get the number of slices needed for one direction and the overlap."""
        pad = int(pad) if pad is not None else 20  # at least some padding needed
        img_size += pad

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
        patch_shape: Tuple[int, int],
        img_size: Tuple[int, int],
        pad: int = None,
    ) -> Tuple[Dict[str, slice], int, int]:
        """Get all the overlapping slices in a dictionary and the needed paddings."""
        y_end, x_end = patch_shape
        nrows, pady = self._get_margins(y_end, img_size[0], stride, pad=pad)
        ncols, padx = self._get_margins(x_end, img_size[1], stride, pad=pad)

        xyslices = []
        for row in range(nrows):
            for col in range(ncols):
                y_start = row * stride
                y_end = y_start + patch_shape[0]
                x_start = col * stride
                x_end = x_start + patch_shape[1]
                xyslices.append((slice(y_start, y_end), slice(x_start, x_end)))

        return xyslices, pady, padx

    def _predict(
        self, x: torch.Tensor, output_shape: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Infer one batch of images."""
        slices, pady, padx = self._get_slices(
            self.stride, self.patch_shape, tuple(x.shape[2:]), self.padding
        )

        padx, modx = divmod(padx, 2)
        pady, mody = divmod(pady, 2)
        padx += modx
        pady += mody

        x = F.pad(x, pad=(padx, padx, pady, pady), mode="reflect")

        # initialize the output masks.
        out_maps = {}
        max_channels = 0
        for head_name, out_channels in self.out_heads:
            # use the largest out channel for recovery mask.
            max_channels = out_channels if out_channels > max_channels else max_channels

            out_maps[head_name] = torch.zeros(
                x.shape[0],
                out_channels,
                *x.shape[2:],
                dtype=x.dtype,
                device=self.device,
            )

        # patches are added to the out mask so need a recovery mask.
        recovery = torch.zeros(
            x.shape[0],
            max_channels,
            *x.shape[2:],
            dtype=x.dtype,
            device=self.device,
        )

        # run inference with the slices
        for yslice, xslice in slices:
            x_i = x[..., yslice, xslice]
            logits = self.model(x_i)

            probs = {}
            for k, out in logits.items():
                apply_boundary_weights = self.head_kwargs[k]["apply_weights"]
                activation = self.head_kwargs[k]["act"]

                if apply_boundary_weights:
                    out *= self.weight_mat

                if activation == "softmax":
                    out = torch.softmax(out, dim=1)
                elif activation == "sigmoid":
                    out = torch.sigmoid(out)
                elif activation == "tanh":
                    out = torch.tanh(out)

                probs[k] = out

            for k, out_map in out_maps.items():
                out_map[..., yslice, xslice] += probs[k]

            recovery[..., yslice, xslice] += 1

        for k, out_map in out_maps.items():
            out = out_map / recovery[:, 0 : out_map.shape[1], ...]
            out_maps[k] = out[..., pady:-pady, padx:-padx]

        # interpolate to the `output_shape`
        if output_shape is not None:
            for k, out_map in out_maps.items():
                out_maps[k] = F.interpolate(
                    out_map, size=output_shape, mode="bilinear", align_corners=False
                )

        return out_maps
