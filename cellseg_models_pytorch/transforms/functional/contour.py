"""
Ported from hovernet repo with minor mods.

MIT License

Copyright (c) 2020 vqdang

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

import numpy as np
import skimage.morphology as morph

__all__ = ["gen_contour_maps"]


def gen_contour_maps(inst_map: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Compute contours for every label object.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        thickness : int, default=1
            Thicnkness of the contour line.

    Returns
    -------
        np.ndarray:
            Contours of the labelled objects. Shape (H, W).
    """
    contour_map = np.zeros_like(inst_map, np.uint8)
    disk = morph.disk(thickness)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        inner = morph.erosion(inst, disk)
        outer = morph.dilation(inst, disk)
        contour_map += outer - inner

    contour_map[contour_map > 0] = 1  # binarize

    return contour_map.astype(np.float64)
