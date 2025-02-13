"""
Ported from hovernet repo.

https://github.com/vqdang/hover_net/blob/master/models/hovernet/targets.py

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
from scipy import ndimage as ndi

from cellseg_models_pytorch.utils import bounding_box, remove_small_objects

__all__ = ["gen_hv_maps"]


def gen_hv_maps(inst_map: np.ndarray, min_size: int = 5) -> np.ndarray:
    """Generate horizontal and vertical gradient maps from label mask.

    Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue
    histology images, Simon Graham, Quoc Dang Vu, Shan E Ahmed Raza, Ayesha Azam,
    Yee Wah Tsang, Jin Tae Kwak, Nasir Rajpoot, Medical Image Analysis, Volume 58,
    2019, 101563, ISSN 1361-8415, doi: https://doi.org/10.1016/j.media.2019.101563.

    https://www.sciencedirect.com/science/article/pii/S1361841519301045

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        min_size : int, default=5
            Min size for objects. Objects less than this many pixels
            are removed.

    Returns
    -------
        np.ndarray:
            The Y- and X-gradient maps in this order. Shape (2, H, W).
            Dtype: float64.
    """
    x_map = np.zeros_like(inst_map, dtype=np.float64)
    y_map = np.zeros_like(inst_map, dtype=np.float64)

    # inst_map = remove_small_objects(inst_map, min_size=min_size, out=inst_map)
    inst_map = remove_small_objects(inst_map, min_size=min_size)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.int32)
        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst = inst[y1:y2, x1:x2]

        # instance center of mass, rounded to nearest pixel
        inst_com = list(ndi.center_of_mass(inst))
        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = np.arange(1, inst.shape[1] + 1)
        inst_y_range = np.arange(1, inst.shape[0] + 1)

        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst == 0] = 0
        inst_y[inst == 0] = 0
        inst_x = inst_x.astype(np.float64)
        inst_y = inst_y.astype(np.float64)

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[y1:y2, x1:x2]
        x_map_box[inst > 0] = inst_x[inst > 0]

        y_map_box = y_map[y1:y2, x1:x2]
        y_map_box[inst > 0] = inst_y[inst > 0]

    hover_map = np.stack([y_map, x_map])

    return hover_map
