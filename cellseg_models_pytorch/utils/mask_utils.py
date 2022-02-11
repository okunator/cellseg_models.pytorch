"""
Some of these functions adapted from the HoVer-Net repo.

https://github.com/vqdang/hover_net/blob/master/

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

from typing import List

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology as morph

__all__ = [
    "remove_small_objects",
    "binarize",
    "fix_duplicates",
    "remove_1px_boundary",
    "remap_label",
    "center_crop",
    "bounding_box",
    "get_inst_types",
    "get_inst_centroid",
    "get_type_instances",
    "one_hot",
    "type_map_flatten",
    "soft_type_flatten",
    "remove_debris_binary",
    "remove_debris_instance",
    "remove_debris_semantic",
    "fill_holes_semantic",
    "label_semantic",
]


# From https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/misc.py
# warning removed
def remove_small_objects(
    ar: np.ndarray,
    min_size: int = 64,
    connectivity: int = 1,
    in_place: bool = False,
    *,
    out: np.ndarray = None,
):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type
        is int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used
        during labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy. Deprecated since version 0.19. Please
        use `out` instead.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or
        string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
           [ True,  True,  True, False, False],
           [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True
    """
    if out is not None:
        in_place = False

    if in_place:
        out = ar
    elif out is None:
        out = ar.copy()

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def binarize(inst_map: np.ndarray) -> np.ndarray:
    """Binarize a labelled instance map.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            Binary mask. Shape (H, W). Type: uint8.
    """
    binary = np.copy(inst_map > 0)
    return binary.astype("uint8")


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def fix_duplicates(inst_map: np.ndarray) -> np.ndarray:
    """Re-label duplicated instances in an instance labelled mask.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            The instance labelled mask without duplicated indices.
            Shape (H, W).
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = ndi.label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map


# ported from https://github.com/vqdang/hover_net/blob/master/src/loader/augs.py
def remove_1px_boundary(inst_map: np.ndarray) -> np.ndarray:
    """Remove 1px around object instances.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns:
    -----------
        np.ndarray:
            The instance labelled mask with 1px of instance boundaries
            removed. Shape (H, W).
    """
    new_inst_map = np.zeros(inst_map.shape[:2], np.int32)
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0)  # 0 is background
    k = morph.disk(1)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        inst = cv2.erode(inst, k, iterations=1)
        new_inst_map[inst > 0] = inst_id

    return new_inst_map


def center_crop(img: np.ndarray, ch: int, cw: int) -> np.ndarray:
    """Center crop an input image.

    Parameters
    ----------
        img : np.ndarray
            Input img. Shape (H, W).
        ch : int
            Crop height.
        cw : int
            Crop width.

    Returns
    -------
        np.ndarray:
            Center cropped image. Shape (ch, cw).
    """
    if len(img.shape) == 3:
        H, W, _ = img.shape
    else:
        H, W = img.shape

    x = W // 2 - (cw // 2)
    y = H // 2 - (ch // 2)

    if len(img.shape) == 3:
        img = img[y : y + ch, x : x + cw, :]
    else:
        img = img[y : y + ch, x : x + cw]

    return img


# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def bounding_box(inst_map: np.ndarray) -> List[int]:
    """Bounding box coordinates for an instance that is given as input.

    This assumes that the `inst_map` has only one instance in it.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        List[int]:
            List of the origin- and end-point coordinates of the bbox.
    """
    rows = np.any(inst_map, axis=1)
    cols = np.any(inst_map, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1

    return [rmin, rmax, cmin, cmax]


# ported from https://github.com/vqdang/hover_net/tree/master/src/metrics/sample
def remap_label(inst_map: np.ndarray) -> np.ndarray:
    """Rename all instance id so that the id is contiguous.

    I.e [0, 1, 2, 3] not [0, 2, 4, 6].

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            Instance labelled mask with remapped contiguous labels.
    """
    inst_list = list(np.unique(inst_map))
    inst_list.remove(0)
    if len(inst_list) == 0:
        return inst_map  # no label

    new_inst_map = np.zeros(inst_map.shape, np.int32)
    for idx, inst_id in enumerate(inst_list):
        new_inst_map[inst_map == inst_id] = idx + 1

    return new_inst_map


# Ported from https://github.com/vqdang/hover_net/blob/master/src/misc/utils.py
def get_inst_centroid(inst_map: np.ndarray) -> np.ndarray:
    """Get centroid x, y coordinates from each unique nuclei instance.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns:
    ----------
        np.ndarray:
            An array of shape (num_instances, 2).
    """
    inst_centroid_list = []
    inst_id_list = list(np.unique(inst_map))
    for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
        mask = np.array(inst_map == inst_id, np.uint8)
        inst_moment = cv2.moments(mask)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid_list.append(inst_centroid)

    return np.array(inst_centroid_list)


def get_inst_types(inst_map: np.ndarray, type_map: np.ndarray) -> np.ndarray:
    """Get the each instance type of a instance labelled mask.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        type_map : np.ndarray
            Type labelled mask. Shape (H, W).

    Returns
    -------
        An np.ndarray of shape (num_instances, 1).
    """
    inst_ids = list(np.unique(inst_map))

    if 0 in inst_ids:
        inst_ids.remove(0)

    inst_types = np.full((len(inst_ids), 1), 0, dtype=np.int32)
    for j, id_ in enumerate(inst_ids):
        inst_type = np.unique(type_map[inst_map == id_])[0]
        inst_types[j] = inst_type

    return inst_types


def get_type_instances(
    inst_map: np.ndarray, type_map: np.ndarray, class_num: int
) -> np.ndarray:
    """Get the instances of the input that belong to class `class_num`.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        type_map : np.ndarray
            Type labelled mask. Shape (H, W).
        class_num : int
            Class label.

    Returns
    -------
        np.ndarray:
            An array  of shape (H, W) where the values equalling
            `class_num` are dropped.
    """
    t = type_map.astype("uint8") == class_num
    imap = np.copy(inst_map)
    imap[~t] = 0

    return imap


def one_hot(type_map: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a type labelled mask of shape (H, W) to one hot (H, W, C).

    Parameters
    ----------
        type_map : np.ndarray
            Type labelled mask. Shape (H, W).
        num_classes (int):
            Number of classes in the dataset.

    Raises:
    -------
        ValueError:
            If the given `num_classes` is less than the observed number
            of classes

    Returns:
    -----------
        np.ndarray:
            An array of the input array (H, W) in one hot format.
            Shape: (H, W, num_classes). Dtype: float64
    """
    ntypes = len(np.unique(type_map)) - 1
    if num_classes < ntypes:
        raise ValueError(
            f"""`num_classes` is less than observed number of classes.
            The type map contains {ntypes}. Got `num_classes`: {num_classes}"""
        )

    return np.eye(num_classes + 1)[type_map]


def type_map_flatten(type_map: np.ndarray) -> np.ndarray:
    """Inverted one-hot.

    Converts a one hot type map of shape (H, W, C) to a single channel
    indice map of shape (H, W).

    Parameters
    ----------
        type_map : np.ndarray
            Type labelled mask. Shape (H, W, C).

    Returns
    -------
        np.ndarray:
            Flattened one hot np.ndarray. Shape (H, W).
    """
    type_out = np.zeros([type_map.shape[0], type_map.shape[1]])
    for t in range(type_map.shape[-1]):
        type_out += type_map[..., t] * t

    return type_out


def soft_type_flatten(type_map: np.ndarray) -> np.ndarray:
    """Flatten a one hot soft mask of shape (H, W, C).

    Parameters
    ----------
        type_map : np.ndarray
            Type labelled mask. Shape (H, W, C).

    Returns
    -------
        np.ndarray:
            Flattened soft mask. Shape (H, W).
    """
    type_out = np.zeros([type_map.shape[0], type_map.shape[1]])
    for i in range(1, type_map.shape[-1]):
        type_tmp = type_map[..., i]
        type_out += type_tmp

    return type_out


def remove_debris_binary(binary_mask: np.ndarray) -> np.ndarray:
    """Take in a binary mask -> fill holes -> removes small objects.

    Parameters
    -----------
        binary_mask : np.ndarray
            A binary mask. Shape (H, W)|(H, W, C).

    Returns
    -------
        np.ndarray:
            Cleaned binary mask of shape (H, W).
    """
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[..., 1]

    out_mask = ndi.binary_fill_holes(binary_mask)
    out_mask = remove_small_objects(binary_mask.astype(bool), min_size=10)

    return out_mask.astype("u1")


def remove_debris_instance(inst_map: np.ndarray, min_size: int = 10):
    """Remove small objects from an inst map.

    (When skimage and ndimage fails)

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).
        min_size : int, default=10
            Min size for the objects that are left untouched.

    Returns
    -------
        np.ndarray:
            Cleaned instance labelled mask of shape (H, W).
    """
    res = np.zeros(inst_map.shape, np.uint32)
    for ix in np.unique(inst_map)[1:]:
        nuc_map = np.copy(inst_map == ix)

        y1, y2, x1, x2 = bounding_box(nuc_map)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        nuc_map_crop = nuc_map[y1:y2, x1:x2].astype("u4")

        nuc_map_crop = remove_small_objects(
            nuc_map_crop.astype(bool), min_size, connectivity=1
        ).astype("u4")

        nuc_map_crop[nuc_map_crop > 0] = ix
        res[y1:y2, x1:x2] += nuc_map_crop

    return res


def remove_debris_semantic(sem_map: np.ndarray, min_size: int = 10000):
    """Remove small objects from a semantic area map.

    Parameters
    ----------
        sem_map : np.ndarray
            Semantic segmentation mask. Shape (H, W).
        min_size : int, default=10000
            Min size for the objects that are left untouched

    Returns
    -------
        np.ndarray:
            Cleaned semantic segmentation mask of shape (H, W).
    """
    res = np.copy(sem_map)
    classes = np.unique(sem_map)

    # skip bg
    if 0 in classes:
        classes = classes[1:]

    for i in classes:

        area = np.array(res == i, np.uint32)
        inst_map = ndi.label(area)[0]
        labels, counts = np.unique(inst_map, return_counts=True)

        for label, npixls in zip(labels, counts):

            if npixls < min_size:
                res[inst_map == label] = 0

                # get the fill label
                y1, y2, x1, x2 = bounding_box(inst_map == label)
                y1 = y1 - 2 if y1 - 2 >= 0 else y1
                x1 = x1 - 2 if x1 - 2 >= 0 else x1
                x2 = x2 + 2 if x2 + 2 <= res.shape[1] - 1 else x2
                y2 = y2 + 2 if y2 + 2 <= res.shape[0] - 1 else y2
                labels, counts = np.unique(res[y1:y2, x1:x2], return_counts=True)

                if 0 in labels and len(labels) > 1:
                    labels = labels[1:]
                    counts = counts[1:]

                fill_label = labels[np.argmax(counts)]
                res[inst_map == label] = fill_label

    return res


def fill_holes_semantic(sem_map: np.ndarray, min_size: int = 5000):
    """Fill holes (background) from a semantic segmentation map.

    Parameters
    ----------
        sem_map : np.ndarray
            Semantic segmentation mask. Shape (H, W).
        min_size : int, default=5000
            Min size for the objects that are left untouched.

    Returns
    -------
        np.ndarray:
            Cleaned semantic segmentation mask of shape (H, W).
    """
    res = np.copy(sem_map)
    bg = res == 0
    bg_objs = ndi.label(bg)[0]

    for i in np.unique(bg_objs)[1:]:
        y1, y2, x1, x2 = bounding_box(bg_objs == i)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= res.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= res.shape[0] - 1 else y2
        crop = res[y1:y2, x1:x2]

        labels, counts = np.unique(crop, return_counts=True)

        if counts[0] > min_size:
            continue

        if len(counts) == 1:
            continue

        # skip 0 index
        labels = labels[1:]
        counts = counts[1:]

        # fill bg objs
        fill_label = labels[np.argmax(counts)]
        crop[crop == 0] = fill_label
        res[y1:y2, x1:x2] = crop

    return res


def label_semantic(sem_map: np.ndarray, sort: bool = True) -> np.ndarray:
    """Labels a given semantic segmentation map.

    Parameters
    ----------
        sem_map : np.ndarray
            Semantic segmentation map. Shape (H, W)
        sort : bool, default=True
            Sort the semantic areas by size in descending order.

    Returns
    -------
        np.ndarray:
            The labelled segmentation map. Shape (H, W).
    """
    sem_inst = np.zeros_like(sem_map)

    counter = 0
    classes, counts = np.unique(sem_map, return_counts=True)

    if 0 in classes:
        classes = classes[1:]
        counts = counts[1:]

    if sort:
        classes = classes[np.argsort(-counts)]

    for c in classes:
        obj_insts = ndi.label(sem_map == c)[0]
        labels = np.unique(obj_insts)

        # If all pixels belong to same (non bg) class, dont slice
        if len(labels) > 1:
            labels = labels[1:]

        for label in labels:
            counter += 1
            obj = obj_insts == label
            sem_inst[obj] += counter

    return sem_inst
