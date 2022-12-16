"""
Ported from the hovernet repo with minor mods.

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
from scipy.ndimage import distance_transform_cdt, distance_transform_edt

from cellseg_models_pytorch.utils import bounding_box

__all__ = ["gen_dist_maps"]


def gen_dist_maps(
    inst_map: np.ndarray, normalize: bool = True, euclidean: bool = True
) -> np.ndarray:
    """Compute distance transforms for every labelled object.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        normalize : bool, default=True
            Normalize the distance maps to [0, 1].
        euclidean : bool, default=True
            If True, uses the euclidean transformation. Else, the chessboard one.

    Returns
    -------
        np.ndarray:
            Distance map of the labelled objects. Shape (H, W).
    """
    dist = np.zeros_like(inst_map, dtype=np.float64)

    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)

        y1, y2, x1, x2 = bounding_box(inst)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

        inst = inst[y1:y2, x1:x2]

        if inst.shape[0] < 2 or inst.shape[1] < 2:
            continue

        # chessboard distance map generation
        # normalize distance to 0-1
        if euclidean:
            inst_dist = distance_transform_edt(inst)
        else:
            inst_dist = distance_transform_cdt(inst)
        inst_dist = inst_dist.astype("float64")

        if normalize:
            max_value = np.amax(inst_dist)
            if max_value <= 0:
                continue
            inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)

        dist_map_box = dist[y1:y2, x1:x2]
        dist_map_box[inst > 0] = inst_dist[inst > 0]

    return dist
