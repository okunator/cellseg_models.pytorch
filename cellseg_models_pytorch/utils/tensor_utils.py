from typing import Union

import numpy as np
import torch

__all__ = ["to_tensor", "to_device"]


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor. Expects HW(C) format."""
    if x.ndim == 2:
        return torch.from_numpy(x).contiguous()
    return torch.from_numpy(x.transpose((2, 0, 1))).contiguous()


def to_device(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Push torch.Tensor or np.ndarray to GPU if it is available.

    Parameters:
        tensor (torch.Tensor or np.ndarray):
            Multi dim array to be pushed to gpu.

    Returns:
        torch.Tensor:
            A tensor. Same shape as input.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor
