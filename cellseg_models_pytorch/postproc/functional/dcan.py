"""
Adapted from HoVer-Net repo (OLD).

https://github.com/vqdang/hover_net/blob/tensorflow-final/src/postproc/other.py

MIT License

Copyright (c) 2018 vqdang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph

from cellseg_models_pytorch.transforms import percentile_normalize99

__all__ = ["post_proc_dcan"]


def post_proc_dcan(
    prob_map: np.ndarray, contour_map: np.ndarray, thresh: float = 0.5, **kwargs
) -> np.ndarray:
    """DCAN post-processing pipeline.

    https://arxiv.org/abs/1604.02677

    Parameters
    ----------
        prob_map : np.ndarray
            Probablilty map. Shape (H, W).
        contour_map : np.ndarray
            Contour map. Shape (H, W).
        thresh : float, default=0.5
            Threshold for the difference between prob_map and contour_map.

    Returns
    -------
        np.ndarray:
            Instance labelled mask. Shape (H, W).
    """
    contour_map = percentile_normalize99(contour_map, amin=-1, amax=1)
    sub = prob_map - contour_map
    pre_insts = ndi.label((sub >= thresh).astype(int))[0]

    inst_ids = np.unique(pre_insts)[1:]
    disk = morph.disk(3)
    inst_map = np.zeros_like(pre_insts)
    for inst_id in inst_ids:
        inst = np.array(pre_insts == inst_id, dtype=np.uint8)
        inst = cv2.dilate(inst, disk, iterations=1)
        inst = ndi.binary_fill_holes(inst)
        inst_map[inst > 0] = inst_id

    return inst_map
