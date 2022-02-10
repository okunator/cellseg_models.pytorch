from typing import Tuple

import numpy as np
from skimage.util import view_as_windows

__all__ = ["_get_margins", "extract_patches", "stitch_patches", "TilerStitcher"]


def _get_margins(
    im_height: int, im_width: int, patch_height: int, patch_width: int, pad: int = 200
) -> Tuple[int, int]:
    """Compute margins for the sliding window patcher.

    Parameters
    ----------
        im_heigth : int
            Height of the original image.
        im_width : int
            Width of the original image.
        patch_height : int
            Height of one image patch.
        patch_width : int
            Width of one image patch.
        pad : int
            Padding in pixels.

    Returns
    -------
        Tuple[int, int]:
            The y- and x-margins in pixels.
    """
    y = np.ceil(((im_height + pad) / patch_height) * patch_height) - im_height
    x = np.ceil(((im_width + pad) / patch_width) * patch_width) - im_width

    return int(y), int(x)


def extract_patches(
    im: np.ndarray, stride: int, patch_shape: Tuple[int, int, int], padding: bool = True
) -> Tuple[np.ndarray, int, int]:
    """Extract overlapping patches from an image.

    Parameters
    ----------
        im : np.ndarray
            Input image. Shape (H, W, C)|(H, W).
        stride : int
            Stride for the sliding window (pixels).
        patch_shape : Tuple[int, int, int]
            Height, width and num of channels of the patches.
        padding : bool
            Use reflection padding to make the input shape divisible
            by the stride size.

    Returns
    -------
        Tuple[np.ndarray, int, int]:
            Patched image of shape (N, H, W, C) and the patch grid
            dimensions `ny` and `nx` that are needed for back-stitching.

    Raises
    ------
        ValueError
            If Given number of channels in `patch_shape` does not match
            the number of channels in `im`.
    """
    if len(patch_shape) != 3:
        raise ValueError(
            f"`patch_shape` has to be in (H, W, C) format. Got: {patch_shape}."
        )

    if patch_shape[-1] == 1 and im.ndim == 2:
        im = im[..., None]

    if im.shape[-1] != patch_shape[-1]:
        raise ValueError(
            f"""Mismatch in given input shapes.
            patch_shape[-1] != im.shape[-1].
            patch_shape[-1] = {patch_shape[-1]}.
            im.shape[-1] = {im.shape[-1]}.
            """
        )

    if padding:
        pad_y, pad_x = _get_margins(
            im.shape[0], im.shape[1], patch_shape[0], patch_shape[1]
        )

        im = np.pad(im, [(pad_y, pad_y), (pad_x, pad_x), (0, 0)], mode="reflect")

    patches = view_as_windows(im, patch_shape, stride)  # (ny, nx, 1, H, W, C)

    return (
        patches.reshape(-1, *patch_shape),  # (N, H, W, C)
        patches.shape[0],  # ny
        patches.shape[1],  # nx
    )


def stitch_patches(
    patches: np.ndarray,
    orig_shape: Tuple[int, ...],
    ny: int,
    nx: int,
    stride: int,
    padding: bool,
) -> np.ndarray:
    """Stitch patches back to original size.

    'Reverse' operation for `extract_patches`.

    Parameters
    ----------
        patches : np.ndarray
            Patched image. Shape (N, pH, pW, C).
        orig_shape Tuple[int, int, int]:
            Shape of the original image.
        ny : int
            Number of rows in patch grid.
        nx : int
            Number of columns in patch grid.
        stride : int
            Stride of the sliding window that was used for patching.
        padding : bool
            Flag, whether padding was used during patching.

    Returns
    -------
        np.ndarray:
            The patches stitched backed to the original size. Shape (H, W, C).

    """
    pad_y = 0
    pad_x = 0
    if padding:
        pad_y, pad_x = _get_margins(orig_shape[0], orig_shape[1], *patches.shape[1:-1])

    pad = stride
    patches = patches.reshape(ny, nx, *patches.shape[1:])
    patches = patches[:, :, 0:pad, 0:pad, :]
    stitched = np.concatenate(np.concatenate(patches, 1), 1)

    stitched = stitched[pad_x : orig_shape[0] + pad_x, pad_y : orig_shape[1] + pad_y, :]

    return stitched


class TilerStitcher:
    def __init__(
        self,
        im_shape: Tuple[int, ...],
        patch_shape: Tuple[int, ...],
        stride: int,
        padding: bool = True,
    ) -> None:
        """
        Image patcher-stitcher.

        Parameters
        ----------
            im_shape : Tuple[int, ...]
                Input image shape (H, W)|(H, W, C).
            patch_shape : Tuple[int, ...]
                Shape of a patch (pH, pW) | (pH, pW, C).
            stride : int
                Stride size for the sliding window.
            padding : bool, default=True
                Use reflection padding (To get a square image.). If set to False,
                the `backstitch` is unlikely able to recover the full sized original
                image but instead cropped image.

        Example
        -------
        >>> im = read_img("/<path>/<to>/img.png)
        >>> print(im.shape)
        (1153, 1307, 3)
        >>> tiler = TilerStitcher(
                im_shape=im.shape,
                patch_shape=(256, 256, 3),
                stride=80,
                padding=True
            )
        >>> patches = tiler.patch(im)
        >>> print(patches.shape)
        (323, 256, 256, 3)
        >>> orig = tiler.backstitch(patches)
        >>> print(orig.shape)
        (1153, 1307, 3)
        """
        self.im_shape = im_shape
        self.patch_shape = patch_shape
        self.stride = stride
        self.padding = padding
        self.nx = None
        self.ny = None

    def patch(self, im: np.ndarray) -> np.ndarray:
        """Extract patches from an input image.

        Parameters
        ----------
            im : np.ndarray
                Input image. Shape: (H, W)|(H, W, C).
        Returns
        -------
            np.ndarray:
                Patched input image. Shape: (N, pH, pW, C).
        """
        patches, self.ny, self.nx = extract_patches(
            im, self.stride, self.patch_shape, self.padding
        )

        return patches

    def backstitch(self, patches: np.ndarray) -> np.ndarray:
        """Stitch patches back to the original image size.

        Parameters
        ----------
            patches : np.ndarray
                Patched image. Shape (N, pH, pW, C).

        Returns
        -------
            np.ndarray:
                Backstitched input image. Shape: (H, W, C).
        """
        stitched = stitch_patches(
            patches, self.im_shape, self.ny, self.nx, self.stride, self.padding
        )
        return stitched.squeeze()
