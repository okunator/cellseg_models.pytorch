from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image

from cellseg_models_pytorch.decoders.multitask_decoder import (
    SoftInstanceOutput,
    SoftSemanticOutput,
)
from cellseg_models_pytorch.utils.convolve import filter2D, gaussian_kernel2d

__all__ = ["Predictor"]

_PredictionMaps = dict[str, SoftInstanceOutput | SoftSemanticOutput | None]


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix.

    The matrix assigns bigger weight on pixels in the center and less weight to pixels
    on the image boundary. Helps dealing with prediction artifacts on tile boundaries.

    Ported from: pytorch-toolbelt

    Args:
        width: Tile width.
        height: Tile height.

    Returns:
        np.ndxay:
            Weight matrix. Shape (H, W).
    """
    xc = width * 0.5
    yc = height * 0.5
    xl = 0
    xr = width
    yb = 0
    yt = height
    Dc = np.zeros((width, height))
    De = np.zeros((width, height))

    Dcx = np.square(np.arange(width) - xc + 0.5)
    Dcy = np.square(np.arange(height) - yc + 0.5)
    Dc = np.sqrt(Dcx[np.newaxis].transpose() + Dcy)

    De_l = np.square(np.arange(width) - xl + 0.5) + np.square(0.5)
    De_r = np.square(np.arange(width) - xr + 0.5) + np.square(0.5)
    De_b = np.square(0.5) + np.square(np.arange(height) - yb + 0.5)
    De_t = np.square(0.5) + np.square(np.arange(height) - yt + 0.5)

    De_x = np.sqrt(np.minimum(De_l, De_r))
    De_y = np.sqrt(np.minimum(De_b, De_t))
    De = np.minimum(De_x[np.newaxis].transpose(), De_y)

    alpha = (width * height) / np.sum(np.divide(De, np.add(Dc, De)))
    W = alpha * np.divide(De, np.add(Dc, De))
    return W


class BasePredictor:
    def __init__(
        self,
        model: nn.Module,
        mixed_precision: bool = True,
    ) -> None:
        """Base class for the predictor."""
        self.model = model
        self.device = next(model.parameters()).device
        self.mixed_precision = mixed_precision

    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray, Image],
        apply_boundary_weight: bool = False,
        save_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the input through the model.

        Note:
            No post-processing is applied.

        Args:
            x: Input image (H, W, C) or input image batch (B, C, H, W).
            apply_boundary_weight:
                Whether to apply boundary weights to mitigate boundary artefacts
                in aux predictions.
                Defaults to True.
            save_intermediate:
                Whether to save intermediate results (logits). If True, the method
                returns a tuple (final predictions, intermediate results), where the
                intermediate results are the raw model outputs before argmax.
                Defaults to False.

        Returns:
            Dict[str, torch.Tensor]:
                Dictionary containing the model predictions (probabilities).
                Shapes: (B, C, H, W).
        """
        # check if the input is a tensor
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x).to(self.device)

        if x.ndim != 4:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected input tensor to have 3 or 4 dimensions (C, H, W) or (B, C, H, W), but got {x.ndim}."
                )

        # create the boundary weight matrix
        if apply_boundary_weight:
            weight_mat = compute_pyramid_patch_weight_loss(x.shape[2], x.shape[3])
            self.weight_mat = (
                torch.from_numpy(weight_mat)
                .float()
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        intermediate = None
        with torch.no_grad():
            if self.mixed_precision:
                with torch.autocast(self.device.type, dtype=torch.float16):
                    probs = self._predict(x)
                    if save_intermediate:
                        intermediate = probs
                    probs = self._argmax(probs)
            else:
                probs = self._predict(x)
                if save_intermediate:
                    intermediate = probs
                probs = self._argmax(probs)

        if save_intermediate:
            return probs, intermediate
        return probs

    def _to_tensor(self, x: Union[np.ndarray, Image]) -> torch.Tensor:
        """Convert to input tensor."""
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = x[:, :, None]

            img = torch.from_numpy(x.transpose((2, 0, 1))).contiguous()
        elif isinstance(x, Image):
            img = torch.as_tensor(np.array(x, copy=True))
            img = img.view(x.size[1], x.size[0], self._get_pil_num_channels(x))
            img = img.permute((2, 0, 1))
        else:
            raise TypeError(
                f"Unsupported type {type(x)}. Expected numpy array or PIL image."
            )

        if isinstance(img, torch.ByteTensor):
            img = img.to(dtype=torch.get_default_dtype()).div(255)

        return img

    def _get_pil_num_channels(self, img: Image) -> int:
        """Get the number of channels in the image."""
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels

    def _smooth_probmap(
        self, x: torch.Tensor, window_size: int = 31, sigma: int = 26
    ) -> torch.Tensor:
        """Apply gaussian filter to smooth probability map."""
        kernel = gaussian_kernel2d(
            window_size=window_size,
            sigma=sigma,
            n_channels=x.shape[1],
            device=x.device,
            dtype=x.dtype,
        )
        return filter2D(x, kernel)

    def _weight_and_activate(
        self,
        x: torch.Tensor,
        activate: bool,
        apply_boundary_weight: bool,
    ) -> torch.Tensor:
        """Activate and optional resizing of the output map."""
        if apply_boundary_weight:
            x = x * self.weight_mat

        if activate:
            x = torch.softmax(x, dim=1)

        return x

    def _argmax(self, probs: Dict[str, Any]) -> torch.Tensor:
        """Apply argmax to the input tensor."""
        probs["nuc"].type_map = probs["nuc"].type_map.argmax(1)
        if probs["cyto"] is not None:
            probs["cyto"].type_map = probs["cyto"].type_map.argmax(1)
        if probs["tissue"] is not None:
            probs["tissue"].type_map = probs["tissue"].type_map.argmax(1)

        return probs

    def _predict(
        self, x: torch.Tensor, apply_boundary_weight: bool = False
    ) -> Dict[str, Any]:
        """Predict the soft masks."""
        soft_masks = self.model(x)

        if soft_masks["nuc"] is not None:
            soft_masks["nuc"].aux_map = self._weight_and_activate(
                soft_masks["nuc"].aux_map,
                activate=False,
                apply_boundary_weight=apply_boundary_weight,
            )
            soft_masks["nuc"].type_map = self._weight_and_activate(
                soft_masks["nuc"].type_map,
                activate=True,
                apply_boundary_weight=False,
            )

            if soft_masks["nuc"].binary_map is not None:
                soft_masks["nuc"].binary_map = self._weight_and_activate(
                    soft_masks["nuc"].binary_map,
                    activate=False,
                    apply_boundary_weight=apply_boundary_weight,
                )

        if soft_masks["cyto"] is not None:
            soft_masks["cyto"].aux_map = self._weight_and_activate(
                soft_masks["cyto"].aux_map,
                activate=False,
                apply_boundary_weight=apply_boundary_weight,
            )
            soft_masks["cyto"].type_map = self._weight_and_activate(
                soft_masks["cyto"].type_map,
                activate=True,
                apply_boundary_weight=False,
            )

            if soft_masks["cyto"].binary_map is not None:
                soft_masks["cyto"].binary_map = self._weight_and_activate(
                    soft_masks["cyto"].binary_map,
                    activate=False,
                    apply_boundary_weight=apply_boundary_weight,
                )

        if soft_masks["tissue"] is not None:
            tissue_prob = self._weight_and_activate(
                soft_masks["tissue"].type_map,
                activate=True,
                apply_boundary_weight=False,
            )
            soft_masks["tissue"].type_map = self._smooth_probmap(tissue_prob)

        return soft_masks


class Predictor(BasePredictor):
    def __init__(
        self,
        model: nn.Module,
        mixed_precision: bool = True,
    ) -> None:
        """Predictor-class for running inference (no post-procesing)."""
        super().__init__(model, mixed_precision=mixed_precision)

    def predict_sliding_win(
        self,
        x: torch.Tensor | np.ndarray | Image,
        window_size: tuple[int, int],
        stride: int,
        padding: int | None = 20,
        apply_boundary_weight: bool = True,
        save_intermediate: bool = False,
    ) -> _PredictionMaps | tuple[_PredictionMaps, _PredictionMaps]:
        """Predict overlapping windows and restore maps to the input image grid.

        Args:
            x: HWC NumPy/PIL image or CHW/BCHW tensor, preprocessed for the
                checkpoint. Tensor inputs must already be on the model device.
            window_size: Height and width of the window size.
            stride: The amount of stride for sliding window.
            padding: Minimum total padding per axis, split between both sides.
                Extra padding completes the windows. None uses the default, 20.
            apply_boundary_weight:
                Downweight auxiliary and binary instance maps at window edges
                before averaging overlaps.
                Defaults to True.
            save_intermediate:
                Also return the intermediate output dictionary.
                Defaults to False.

        Returns:
            Full-image maps grouped by head. Type maps contain class IDs with
            shape BHW; auxiliary and binary maps retain their BCHW channels.
            With save_intermediate=True, returns a pair of output dictionaries.

        Raises:
            ValueError: Window dimensions or stride are nonpositive, stride leaves
                gaps, padding is negative, or the input image has invalid dimensions.
        """
        if len(window_size) != 2 or any(size <= 0 for size in window_size):
            raise ValueError("window_size must contain two positive dimensions.")
        if stride <= 0 or stride > min(window_size):
            raise ValueError(
                "stride must be positive and no larger than either window dimension."
            )
        padding = 20 if padding is None else padding
        if padding < 0:
            raise ValueError("padding must be nonnegative.")

        # check if the input is a tensor
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x).to(self.device)

        if x.ndim != 4:
            if x.ndim == 3:
                x = x.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected input tensor to have 3 or 4 dimensions (C, H, W) or (B, C, H, W), but got {x.ndim}."
                )
        if min(x.shape[-2:]) <= 0:
            raise ValueError("Input image must have positive height and width.")

        if apply_boundary_weight:
            weight_mat = compute_pyramid_patch_weight_loss(*window_size)
            self.weight_mat = (
                torch.from_numpy(weight_mat)
                .float()
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        intermediate = None
        with torch.no_grad():
            if self.mixed_precision:
                with torch.autocast(self.device.type, dtype=torch.float16):
                    probs = self._predict_sliding_win(
                        x, window_size, stride, padding, apply_boundary_weight
                    )
                    if save_intermediate:
                        intermediate = probs
                    probs = self._argmax(probs)
            else:
                probs = self._predict_sliding_win(
                    x, window_size, stride, padding, apply_boundary_weight
                )
                if save_intermediate:
                    intermediate = probs
                probs = self._argmax(probs)

        if save_intermediate:
            return probs, intermediate

        return probs

    @staticmethod
    def _get_margins(
        first_endpoint: int, img_size: int, stride: int, pad: int | None = None
    ) -> tuple[int, int]:
        """Return the window count and total padding needed along one image axis."""
        pad = int(pad) if pad is not None else 20
        n = max(0, (img_size + pad - first_endpoint + stride - 1) // stride) + 1
        return n, first_endpoint + (n - 1) * stride - img_size

    @staticmethod
    def _get_slices(
        stride: int,
        patch_shape: tuple[int, int],
        img_size: tuple[int, int],
        pad: int | None = None,
    ) -> tuple[list[tuple[slice, slice]], int, int]:
        """Return window row/column slices and total vertical/horizontal padding."""
        y_end, x_end = patch_shape
        nrows, pady = Predictor._get_margins(y_end, img_size[0], stride, pad=pad)
        ncols, padx = Predictor._get_margins(x_end, img_size[1], stride, pad=pad)

        xyslices = []
        for row in range(nrows):
            for col in range(ncols):
                y_start = row * stride
                y_end = y_start + patch_shape[0]
                x_start = col * stride
                x_end = x_start + patch_shape[1]
                xyslices.append((slice(y_start, y_end), slice(x_start, x_end)))

        return xyslices, pady, padx

    def _zeros(
        self, B: int, C: int, H: int, W: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(B, C, H, W, dtype=dtype, device=device)

    def _init_output(
        self, x: torch.Tensor, soft_masks: SoftInstanceOutput | SoftSemanticOutput
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Allocate full-image type, auxiliary and binary maps for one output head.

        Args:
            x: Padded BCHW input; supplies batch size, image size, dtype and device.
            soft_masks: One window's output; supplies the head's channel counts.

        Returns:
            Type, auxiliary and binary buffers. Absent maps return None.
        """
        C = soft_masks.type_map.shape[1]
        out = self._zeros(x.shape[0], C, *x.shape[2:], x.dtype, x.device)

        out_aux = None
        if isinstance(soft_masks, SoftInstanceOutput):
            Ca = soft_masks.aux_map.shape[1]
            out_aux = self._zeros(x.shape[0], Ca, *x.shape[2:], x.dtype, x.device)

        out_binary = None
        if soft_masks.binary_map is not None:
            Cb = soft_masks.binary_map.shape[1]
            out_binary = self._zeros(x.shape[0], Cb, *x.shape[2:], x.dtype, x.device)

        return out, out_aux, out_binary

    def _predict_sliding_win(
        self,
        x: torch.Tensor,
        window_size: tuple[int, int],
        stride: int,
        padding: int = 20,
        apply_boundary_weight: bool = True,
    ) -> _PredictionMaps:
        """Combine window predictions into maps aligned with the unpadded BCHW input."""
        height, width = x.shape[-2:]
        slices, pady, padx = self._get_slices(
            stride, window_size, tuple(x.shape[2:]), padding
        )

        top, left = pady // 2, padx // 2
        bottom, right = pady - top, padx - left
        # Reflection cannot extend beyond a small input; repeat its edge instead.
        mode = (
            "reflect"
            if max(top, bottom) < height and max(left, right) < width
            else "replicate"
        )
        x = F.pad(x, pad=(left, right, top, bottom), mode=mode)
        recovery = self._zeros(1, 1, *x.shape[2:], x.dtype, x.device)
        aux_recovery = torch.zeros_like(recovery) if apply_boundary_weight else recovery
        for i, (yslice, xslice) in enumerate(slices):
            x_i = x[..., yslice, xslice]
            soft_masks = self._predict(x_i, apply_boundary_weight=apply_boundary_weight)

            # create output matrices
            if i == 0:
                nuc, nuc_aux, nuc_binary = self._init_output(x, soft_masks["nuc"])
                if soft_masks["cyto"] is not None:
                    cyto, cyto_aux, cyto_binary = self._init_output(
                        x, soft_masks["cyto"]
                    )
                if soft_masks["tissue"] is not None:
                    tissue, _, tissue_binary = self._init_output(
                        x, soft_masks["tissue"]
                    )

            # add the soft masks to the output matrices
            nuc[..., yslice, xslice] += soft_masks["nuc"].type_map
            nuc_aux[..., yslice, xslice] += soft_masks["nuc"].aux_map
            if nuc_binary is not None:
                nuc_binary[..., yslice, xslice] += soft_masks["nuc"].binary_map

            if soft_masks["cyto"] is not None:
                cyto[..., yslice, xslice] += soft_masks["cyto"].type_map
                cyto_aux[..., yslice, xslice] += soft_masks["cyto"].aux_map
                if cyto_binary is not None:
                    cyto_binary[..., yslice, xslice] += soft_masks["cyto"].binary_map

            if soft_masks["tissue"] is not None:
                tissue[..., yslice, xslice] += soft_masks["tissue"].type_map
                if tissue_binary is not None:
                    tissue_binary[..., yslice, xslice] += soft_masks[
                        "tissue"
                    ].binary_map

            recovery[..., yslice, xslice] += 1
            if apply_boundary_weight:
                aux_recovery[..., yslice, xslice] += self.weight_mat

        rows, cols = slice(top, top + height), slice(left, left + width)
        # Auxiliary/binary instance maps were weighted; divide by those same weights.
        soft_masks["nuc"].type_map = (nuc / recovery)[..., rows, cols]
        soft_masks["nuc"].aux_map = (nuc_aux / aux_recovery)[..., rows, cols]
        if nuc_binary is not None:
            soft_masks["nuc"].binary_map = (nuc_binary / aux_recovery)[..., rows, cols]

        if soft_masks["cyto"] is not None:
            soft_masks["cyto"].type_map = (cyto / recovery)[..., rows, cols]
            soft_masks["cyto"].aux_map = (cyto_aux / aux_recovery)[..., rows, cols]
            if cyto_binary is not None:
                soft_masks["cyto"].binary_map = (cyto_binary / aux_recovery)[
                    ..., rows, cols
                ]

        if soft_masks["tissue"] is not None:
            soft_masks["tissue"].type_map = (tissue / recovery)[..., rows, cols]
            if tissue_binary is not None:
                soft_masks["tissue"].binary_map = (tissue_binary / recovery)[
                    ..., rows, cols
                ]

        return soft_masks
