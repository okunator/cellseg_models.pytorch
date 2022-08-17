import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.segmentation as segm

from cellseg_models_pytorch.utils import (
    binarize,
    naive_thresh_prob,
    percentile_normalize99,
    remove_small_objects,
)


def post_proc_dran(
    inst_map: np.ndarray, contour_map: np.ndarray, **kwargs
) -> np.ndarray:
    """DRAN post-processing pipeline.

    https://www.frontiersin.org/articles/10.3389/fbioe.2019.00053/full

    This is not the original implementation but follows along the steps
    introduced in the paper. Added dilation to the end of the pipeline.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled or binary mask. Shape (H, W).
        contour_map : np.ndarray
            Contour map. Shape (H, W).

    Returns
    -------
        np.ndarray:
            post-processed inst map. Shape (H, W)
    """
    contour = percentile_normalize99(contour_map, amin=0, amax=1)
    cnt_binary = binarize(naive_thresh_prob(contour))
    cnt_binary = cv2.dilate(cnt_binary, morph.disk(3), iterations=1)

    binary = binarize(inst_map)
    binary = remove_small_objects(binary.astype(bool), min_size=10).astype("uint8")

    markers = binary - cnt_binary
    markers[markers != 1] = 0
    markers = ndi.label(markers)[0]

    distance = ndi.distance_transform_edt(markers)
    distance = 255 * (distance / (np.amax(distance) + 1e-7))

    inst_map = segm.watershed(-distance, markers=markers, mask=binary)

    return inst_map
