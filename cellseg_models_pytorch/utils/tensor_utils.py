from typing import Union

import numpy as np
import torch

__all__ = ["to_tensor", "to_device", "tensor_one_hot"]


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor. Expects HW(C) format."""
    if x.ndim == 2:
        x = x[:, :, None]

    return torch.from_numpy(x.transpose((2, 0, 1))).contiguous()


def to_device(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Push torch.Tensor or np.ndarray to GPU if it is available.

    Parameters
    ----------
        tensor : torch.Tensor or np.ndarray:
            Multi dim array to be pushed to gpu.

    Returns
    -------
        torch.Tensor:
            A tensor. Same shape as input.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor


def tensor_one_hot(type_map: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert a segmentation mask into one-hot-format.

    I.e. Takes in a segmentation mask of shape (B, H, W) and reshapes it
    into a tensor of shape (B, C, H, W).

    Parameters
    ----------
        type_map : torch.Tensor
            Multi-label Segmentation mask. Shape (B, H, W).
        n_classes : int
            Number of classes. (Zero-class included.)

    Returns
    -------
        torch.Tensor:
            A one hot tensor. Shape: (B, C, H, W). Dtype: torch.FloatTensor.

    Raises
    ------
        TypeError: If input is not torch.int64.
    """
    if not type_map.dtype == torch.int64:
        raise TypeError(
            f"""
            Input `type_map` should have dtype: torch.int64. Got: {type_map.dtype}."""
        )

    one_hot = torch.zeros(
        type_map.shape[0],
        n_classes,
        *type_map.shape[1:],
        device=type_map.device,
        dtype=type_map.dtype,
    )

    return one_hot.scatter_(dim=1, index=type_map.unsqueeze(1), value=1.0) + 1e-7
