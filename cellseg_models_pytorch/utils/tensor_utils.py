from typing import Union

import numpy as np
import torch


def ndarray_to_tensor(
    array: np.ndarray, in_dim_format: str, out_dim_format: str
) -> torch.Tensor:
    """Convert img (H, W)|(H, W, C)|(B, H, W, C) to a tensor.

    Parameters
    ----------
        array : np.ndarray
            Numpy matrix. Shape: (H, W)|(H, W, C)|(B, H, W, C)
        in_dim_format : str
            The order of the dimensions in the input array.
            One of: "HW", "HWC", "BHWC", "BCHW", "BHW"
        out_dim_format : str
            The order of the dimensions in the output tensor.
            One of: "HW", "HWC", "BHWC", "BCHW", "BHW"

    Returns
    -------
        torch.Tensor:
            Input converted to a batched tensor. Shape .

    Raises
    ------
        TypeError:
            If input array is not np.ndarray.
        ValueError:
            If input array has wrong number of dimensions
        ValueError:
            If `in_dim_format` param is illegal.
        ValueError:
            If `out_dim_format` param is illegal.

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input type: {type(array)} is not np.ndarray")

    if not 1 < len(array.shape) <= 4:
        raise ValueError(
            f"ndarray.shape {array.shape}, rank needs to be between [2, 4]"
        )

    dim_types = ("HW", "HWC", "BHWC", "BCHW", "BHW")
    if in_dim_format not in dim_types:
        raise ValueError(
            f"Illegal `in_dim_format`. Got {in_dim_format}. Allowed: {dim_types}"
        )

    if out_dim_format not in dim_types:
        raise ValueError(
            f"Illegal `out_dim_format`. Got {out_dim_format}. Allowed: {dim_types}"
        )

    if not len(array.shape) == len(in_dim_format):
        raise ValueError(
            f"""
            Mismatching input dimensions.
            Input Shape: {array.shape}.
            while `in_dim_format` is set to: {in_dim_format}"""
        )

    if in_dim_format in ("HW", "BHW"):
        if out_dim_format in ("HWC", "BHWC", "BCHW"):
            array = array[..., None]

    if in_dim_format in ("HW", "HWC"):
        if out_dim_format in ("BHWC", "BCHW", "BHW"):
            array = array[None, ...]

    if (
        len(array.shape) == 4
        and in_dim_format in ("BHWC", "HWC", "HW")
        and out_dim_format == "BCHW"
    ):
        array = array.transpose(0, 3, 1, 2)

    if len(array.shape) == 4 and in_dim_format == "BCHW" and out_dim_format == "BHWC":
        array = array.transpose(0, 2, 3, 1)

    return torch.from_numpy(array)


def tensor_to_ndarray(
    tensor: torch.Tensor, in_dim_format: str, out_dim_format: str
) -> np.ndarray:
    """Convert a tensor into a numpy ndarray.

    Parameters
    ----------
        tensor : torch.Tensor
            The input tensor. Shape: (B, H, W)|(B, C, H, W)
        in_dim_format : str
            The order of the dimensions in the input array.
            One of: "BCHW", "BHW"
        out_dim_format : str
            The order of the dimensions in the output tensor.
            One of: "HW", "HWC", "BHWC", "BHW"

    Returns
    -------
        np.ndarray:
            An ndarray. Shape(B, H, W, C)|(B, H, W)|(H, W, C)|(H, W)

    Raises
    ------
        TypeError:
            If input array is not torch.Tensor.
        ValueError:
            If input array has wrong number of dimensions
        ValueError:
            If `in_dim_format` param is illegal.
        ValueError:
            If `out_dim_format` param is illegal.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type: {type(tensor)} is not torch.Tensor")

    if not 3 <= tensor.dim() <= 4:
        raise ValueError(
            f"""
        The input tensor needs to have shape (B, H, W) or (B, C, H, W)."
        Got: {tensor.shape}""",
        )

    in_dim_types = ("BCHW", "BHW")
    if in_dim_format not in in_dim_types:
        raise ValueError(
            f"Illegal `in_dim_format`. Got {in_dim_format}. Allowed: {in_dim_types}"
        )

    out_dim_types = ("BCHW", "BHWC", "BHW", "HWC", "HW")
    if out_dim_format not in out_dim_types:
        raise ValueError(
            f"Illegal `out_dim_format`. Got {out_dim_format}. Allowed: {out_dim_types}"
        )

    # detach and bring to cpu
    array = tensor.detach()
    if tensor.is_cuda:
        array = array.cpu()

    array = array.numpy()
    if array.ndim == 4 and out_dim_format != "BCHW":
        array = array.transpose(0, 2, 3, 1)  # (B, H, W, C)

    if out_dim_format == "HW" and array.ndim == 4:
        array = array.squeeze()

    if out_dim_format == "HWC" and array.ndim == 4:
        try:
            array = array.squeeze(axis=0)
        except Exception:
            pass

    if out_dim_format == "BHW" and array.ndim == 4:
        try:
            array = array.squeeze(axis=-1)
        except Exception:
            pass

    if in_dim_format == "BHW":
        if out_dim_format == "BHWC":
            array = array[..., None]
        elif out_dim_format == "HW":
            array = array.squeeze()

    return array


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
