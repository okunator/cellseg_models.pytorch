from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NORM_LOOKUP, ndarray_to_tensor

__all__ = ["Predictor"]


def compute_pyramid_patch_weight_loss(width: int, height: int) -> np.ndarray:
    """Compute a weight matrix.

    The matrix assigns bigger weight on pixels in the center and less weight to pixels
    on the image boundary. Helps dealing with prediction artifacts on tile boundaries.

    Ported from: pytorch-toolbelt

    Parameters
    ----------
        width : int
            Tile width.
        height : int
            Tile height

    Returns
    -------
        np.ndarray:
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


class Predictor:
    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int],
        normalization: str = None,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        """Predict dense soft masks with this helper class.

        Includes a weight matrix that can assign bigger weight on pixels
        in center and less weight to pixels on image boundary. helps
        dealing with prediction artifacts on tile boundaries.

        Parameters
        ----------
            model : nn.Module:
                nn.Module pytorch model
            patch_size : Tuple[int, int]
                The height and width of the input patch.
            normalization : str, optional
                Apply img normalization (Same as during training). One of "dataset",
                "minmax", "norm", "percentile", None.
            mean : Tuple[float, float, float]
                Means for each channel. Used only if `normalization` == "dataset"
            std : Tuple[float, float, float]
                Stds for each channel. Used only if `normalization` == "dataset"
            device : str or torch.device, default="cuda"
                The device of the model. One of "cpu", "cuda".
        """
        allowed = list(NORM_LOOKUP.keys()) + [None]
        if normalization not in allowed:
            raise ValueError(
                f"Illegal normalization. Got: {normalization}. Allowed: {allowed}"
            )

        self.normalization = normalization
        self.mean = mean
        self.std = std

        self.model = model
        self.model.eval()

        weight_mat = compute_pyramid_patch_weight_loss(patch_size[0], patch_size[1])
        self.weight_mat = (
            torch.from_numpy(weight_mat).float().to(device).unsqueeze(0).unsqueeze(0)
        )

    def forward_pass(
        self, patch: Union[np.ndarray, torch.Tensor], in_dim_format: str = "HWC"
    ) -> Dict[str, torch.Tensor]:
        """Input an image patch or batch of patches to the network and return logits.

        Parameters
        ----------
            patch : np.ndarray or torch.Tensor
                Image patch.
            in_dim_format : str, default="HWC"
                The order of the dimensions in the input array.
                One of: "HWC", "BHWC"


        Returns
        -------
            Dict[str, torch.Tensor]:
                A dictionary of output name mapped to output tensors.
        """
        dim_types = ("HWC", "BHWC")
        if in_dim_format not in dim_types:
            raise ValueError(
                f"Illegal `in_dim_format`. Got {in_dim_format}. Allowed: {dim_types}"
            )

        if isinstance(patch, np.ndarray):
            patch = ndarray_to_tensor(patch, in_dim_format, "BCHW")

        if self.normalization is not None:
            kwargs = {}
            if self.mean is not None or self.std is not None:
                kwargs = {"mean": self.mean, "std": self.std}

            patch = NORM_LOOKUP[self.normalization](patch, **kwargs)
        else:
            patch = patch.float()

        with torch.no_grad():
            out = self.model(patch)

        return out

    def classify(
        self,
        patch: torch.Tensor,
        act: Union[str, None] = "softmax",
        apply_weights: bool = False,
    ) -> np.ndarray:
        """Take in logits and output probabilities.

        Additionally apply a weight matrix to help with boundary artefacts.

        Parameters
        ----------
            patch : torch.Tensor
                A tensor of logits produced by the network. Shape: (B, C, H, W)
            act : str, default="softmax"
                Activation to be used. One of: "sigmoid", "softmax" or None
            apply_weights : bool, default=False
                Apply a weight matrix that assigns bigger weight on pixels in center and
                less weight to pixels on the image boundary.

        Returns
        -------
            np.ndarray:
                The model prediction. Same shape as input `patch`.
        """
        allowed = ("sigmoid", "softmax", "tanh", None)
        if act not in allowed:
            raise ValueError(
                f"Illegal activation func given. Got: {act}. Allowed: {allowed}"
            )

        # Add weights to pred matrix
        if apply_weights:
            # work out the tensor shape first for the weight mat
            B, C = patch.shape[:2]
            W = torch.repeat_interleave(
                self.weight_mat,
                dim=1,
                repeats=C,
            ).repeat_interleave(repeats=B, dim=0)
            patch *= W

        # apply classification activation
        if act == "sigmoid":
            pred = torch.sigmoid(patch)
        elif act == "tanh":
            pred = torch.tanh(patch)
        elif act == "softmax":
            pred = F.softmax(patch, dim=1)
        else:
            pred = patch

        return pred
