import numpy as np
import skimage.filters as filters
import skimage.segmentation as segm

from .mask_utils import remove_debris_binary

__all__ = [
    "naive_thresh_prob",
    "naive_thresh",
    "niblack_thresh",
    "sauvola_thresh",
    "morph_chan_vese_thresh",
    "argmax",
]


def naive_thresh_prob(
    prob_map: np.ndarray, threshold: float = 0.5, **kwargs
) -> np.ndarray:
    """Threshold a sigmoid/softmax activated soft mask.

    Parameters
    ----------
        prob_map : np.ndarray
            Soft mask to be thresholded. Shape (H, W)
        threshold: float, default=0.5
            Thresholding cutoff between [0, 1]

    Returns
    -------
        np.ndarray:
            Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"thresh = {threshold}. given threshold not between [0,1]")

    seg = prob_map.copy()
    seg = seg >= threshold
    seg = remove_debris_binary(seg)

    return seg


def naive_thresh(prob_map: np.ndarray, threshold: int = 2, **kwargs) -> np.ndarray:
    """Threshold a soft mask. Values can be logits or probabilites.

    Parameters
    ----------
        prob_map : np.ndarray
            Soft mask to be thresholded. Shape (H, W).
        threshold : int, default=2
            Value used to divide the max value of the mask.

    Returns
    -------
        np.ndarray:
            Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    seg = prob_map.copy()
    seg[seg < np.amax(prob_map) / threshold] = 0
    seg[seg > np.amax(prob_map) / threshold] = 1
    seg = remove_debris_binary(seg)

    return seg


def niblack_thresh(prob_map: np.ndarray, win_size: int = 13, **kwargs) -> np.ndarray:
    """Do niblack thresholding (skimage wrapper).

    Parameters
    ----------
        prob_map : np.ndarray
            Soft mask to be thresholded. Shape (H, W).
        win_size : int, default=13
            Size of the window used in the thresholding algorithm.

    Returns
    -------
        np.ndarray:
            Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    thresh = filters.threshold_niblack(prob_map, window_size=win_size)
    seg = prob_map > thresh
    seg = remove_debris_binary(seg)

    return seg


def sauvola_thresh(prob_map: np.ndarray, win_size: int = 33, **kwargs) -> np.ndarray:
    """Do sauvola thresholding (skimage wrapper).

    Parameters
    ----------
        prob_map : np.ndarray
            Soft mask to be thresholded. Shape (H, W).
        win_size : int, default=33
            Size of the window used in the thresholding algorithm.

    Returns
    -------
        np.ndarray:
            Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    thresh = filters.threshold_sauvola(prob_map, window_size=win_size)
    seg = prob_map > thresh
    seg = remove_debris_binary(seg)

    return seg


def morph_chan_vese_thresh(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """Morphological chan vese method for thresholding. Skimage wrapper.

    Parameters
    ----------
        prob_map : np.ndarray
            Soft mask to be thresholded. Shape (H, W).

    Returns
    -------
        np.ndarray:
            Thresholded soft mask. Shape: (H, W). Type: uint8.
    """
    init_ls = segm.checkerboard_level_set(prob_map.shape, 2)
    ls = segm.morphological_chan_vese(prob_map, 35, smoothing=1, init_level_set=init_ls)

    hist = np.histogram(ls)[0]
    if hist[-1] > hist[0]:
        ls = 1 - ls

    seg = remove_debris_binary(ls)

    return seg


def argmax(prob_map: np.ndarray, **kwargs) -> np.ndarray:
    """Take argmax of a one_hot logits or prob map.

    Parameters
    ----------
        prob_map : np.ndarray
            The probability map of shape (H, W, C)|(H, W).

    Returns
    -------
        np.ndarray:
            a mask of argmax indices. Shape: (H, W). Type: uint8.
    """
    if len(prob_map.shape) == 2:
        inv_prob = 1 - prob_map
        prob_map = np.stack([inv_prob, prob_map], axis=-1)

    seg = np.argmax(prob_map, axis=-1).astype("u4")

    return seg
