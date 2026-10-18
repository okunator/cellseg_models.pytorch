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
    ) -> _PredictionMaps | tuple[_PredictionMaps, _PredictionMaps]:
        """Predict a single image or a batch without windowing.

        Args:
            x: RGB NumPy/PIL image (H, W, C), or floating tensor (C, H, W) or
                (B, C, H, W). B is batch size, C channels, H height and W width.
                NumPy/PIL uint8 values are divided by 255. Other image values are
                unchanged. Tensors must already have the correct dtype/device.
                Apply the checkpoint's normalization before prediction.
            apply_boundary_weight: Request a center-weighting matrix. In this
                non-windowed path, it is constructed but not applied to predictions.
                Defaults to False.
            save_intermediate: Return an additional reference to the result. The
                current implementation does not preserve a separate raw-score copy.

        Returns:
            A dictionary with ``nuc``, ``cyto`` and ``tissue`` entries; unused
            outputs are None. Instance ``type_map`` values are integer class labels
            (B, H, W); maps for separating objects remain floating tensors (B, channels, H, W).
            Results stay on the network device. With ``save_intermediate=True``,
            returns the same dictionary twice. Call the network directly for raw
            class scores.

        Raises:
            ValueError: The input is not a three- or four-dimensional tensor after
                image conversion.
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
    """Run a configured network on images, including overlapping windows.

    This is the lower-level alternative to a model class's ``predict`` method.
    It returns class labels and reconstruction maps, not individual objects.
    """

    def __init__(
        self,
        model: nn.Module,
        mixed_precision: bool = True,
    ) -> None:
        """Choose the network and prediction precision.

        Args:
            model: A package network that returns the ``nuc``, ``cyto`` and ``tissue``
                map entries. Put it on the intended device and call ``eval()`` first;
                the predictor does not change its training mode.
            mixed_precision: Use float16 autocasting on the network's device.
                False keeps the network's normal computation dtype; use it for
                ordinary float32 CPU prediction.
        """
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
        """Predict overlapping windows and combine them at the original image size.

        Args:
            x: RGB NumPy/PIL image (H, W, C), or floating tensor (C, H, W) or
                (B, C, H, W). B is batch size, C channels, H height and W width.
                NumPy/PIL uint8 values are divided by 255. Other image values are
                unchanged. Tensors must already have the correct dtype/device.
                Apply the checkpoint's normalization before prediction.
            window_size: Window (height, width). Each dimension must be positive and
                compatible with the network's encoder and decoder.
            stride: Positive step in pixels, no larger than either window dimension.
            padding: Extra coverage used to build the window grid. None uses 20.
                Reflection padding is used where possible, otherwise edge replication.
                Padding is removed from the returned maps.
            apply_boundary_weight: Reduce the influence of window edges on maps used
                to separate objects and foreground maps. Class probabilities are averaged before
                selecting class labels.
            save_intermediate: Return an additional reference to the result; it does
                not keep a separate copy of the raw class scores.

        Returns:
            A dictionary with ``nuc``, ``cyto`` and ``tissue`` entries; unused
            outputs are None. Instance ``type_map`` values are integer class labels
            (B, H, W); maps for separating objects remain floating tensors (B, channels, H, W).
            Results stay on the network device. With ``save_intermediate=True``,
            returns the same dictionary twice. Call the network directly for raw
            class scores.

        Raises:
            ValueError: Window sizes, stride, padding or image dimensions are invalid.
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
