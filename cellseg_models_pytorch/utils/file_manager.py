import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import scipy.io as sio

from .mask_utils import (
    bounding_box,
    fix_duplicates,
    get_inst_centroid,
    get_inst_types,
    label_semantic,
)
from .multiproc import run_pool

try:
    import tables as tb

    _has_tb = True
except ModuleNotFoundError:
    _has_tb = False


class FileHandler:
    """Class for handling file I/O."""

    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        """Read an image & convert from bgr to rgb. (cv2 reads imgs in bgr).

        Parameters
        ----------
            path : str or Path
                Path to the image file.

        Returns
        -------
            np.ndarray:
                The image. Shape (H, W, 3).
        """
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def write_img(path: Union[str, Path], img: np.ndarray) -> None:
        """Write an image.

        Parameters
        ----------
            path : str or Path
                Path to the image file.
            img : np.ndarray
                The image to be written.

        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def read_mat(
        path: Union[str, Path],
        key: str = "inst_map",
        retype: bool = True,
        return_all: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray], None]:
        """Read a mask from a .mat file.

        If a mask is not found, return None

        Parameters
        ----------
            path : str or Path
                Path to the .mat file.
            key : str, default="inst_map"
                Name/key of the mask type that is being read from .mat
            retype : bool, default=True
                Convert the matrix type.
            return_all : bool, default=False
                Return the whole dict. Overrides the `key` arg.


        Raises
        ------
            ValueError: If an illegal key is given.

        Returns
        -------
            Union[np.ndarray, List[np.ndarray], None]:
                if return_all == False:
                    The instance/type/semantic labelled mask. Shape: (H, W).
                if return_all == True:
                    All the masks in the .mat file returned in a dictionary.
        """
        dtypes = {
            "inst_map": "int32",
            "type_map": "int32",
            "sem_map": "int32",
            "inst_centroid": "float64",
            "inst_type": "int32",
        }

        path = Path(path)
        if not path.exists():
            raise ValueError(f"{path} not found")

        try:
            mask = sio.loadmat(path.as_posix())
        except Exception:
            mask = None

        if not return_all:
            allowed = ("inst_map", "type_map", "inst_centroid", "inst_type", "sem_map")
            if key not in allowed:
                raise ValueError(f"Illegal key given. Got {key}. Allowed: {allowed}")

            try:
                mask = mask[key]
                if retype:
                    mask = mask.astype(dtypes[key])
            except Exception:
                mask = None

        return mask

    @staticmethod
    def get_geo_obj(what: str = "qupath") -> Dict[str, str]:
        """Get the dict format for a geojson obj.

        For example: get the obj in QuPath PathCellDetection obj

        Parameters
        ----------
            what : str
                One of "qupath", "simple"

        Returns
        -------
            Dict[str, Any]:
                A dictionary in geojson format.
        """
        allowed = ("qupath",)
        if what not in allowed:
            raise ValueError(f"Illegal `what`-arg. Got: {what}. Allowed: {allowed}")

        geo_obj = {}
        if what == "qupath":
            geo_obj.setdefault("type", "Feature")

            # PathCellAnnotation, PathCellDetection, PathDetectionObject
            geo_obj.setdefault("id", "PathCellDetection")
            geo_obj.setdefault("geometry", {"type": "Polygon", "coordinates": None})
            geo_obj.setdefault(
                "properties",
                {
                    "isLocked": "false",
                    "measurements": [],
                    "classification": {"name": None},
                },
            )

        return geo_obj

    @staticmethod
    def get_gson(
        inst: np.ndarray,
        type: np.ndarray,
        classes: Dict[str, int],
        soft_type: np.ndarray = None,
        x_offset: int = 0,
        y_offset: int = 0,
        geo_format: str = "qupath",
    ) -> Dict[str, Any]:
        """Get the labels in geojson format.

        Parameters
        ----------
            inst : np.ndarray
                Instance labelled mask. Shape: (H, W).
            type : np.ndarray
                Cell type labelled semantic segmentation mask. Shape: (H, W).
            classes : Dict[str, int]
                Class dict e.g. {"inflam":1, "epithelial":2, "connec":3}
            soft_type : np.ndarray, default=None
                Softmax type mask. Shape: (C, H, W). C is the number of classes.
            x_offset : int, default=0
                x-coordinate offset. (to set geojson to .mrxs wsi coordinates)
            y_offset : int, default=0
                y-coordinate offset. (to set geojson to .mrxs wsi coordinates)
            geo_format : str, default="qupath"
                The format for the geo object. "qupath" format allows the result file
                to be read with QuPath. "simple" format allows for geopandas etc.

        Returns
        -------
            Dict[str, Any]:
                A geojson dictionary of the instance labelled mask.
        """
        inst_map = fix_duplicates(inst)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        geo_objs = []
        for inst_id in inst_list:
            # set up the annotation geojson obj

            # Get cell instance and cell type
            inst = np.array(inst_map == inst_id, np.uint8)
            inst_type = type[inst_map == inst_id].astype("uint8")
            inst_type = np.unique(inst_type)[0]

            inst_type = [key for key in classes.keys() if classes[key] == inst_type][0]

            # type probabilities
            if soft_type is not None:
                type_probs = soft_type[..., inst_map == inst_id].mean(axis=1)
                inst_type_soft = dict(zip(classes.keys(), type_probs))
                # convert to float for json serialization
                for key in inst_type_soft.keys():
                    inst_type_soft[key] = float(inst_type_soft[key])

            # get the cell contour coordinates
            contours, _ = cv2.findContours(inst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # got a line instead of a polygon
            if contours[0].shape[0] < 3:
                continue

            # shift coordinates based on the offsets
            if x_offset:
                contours[0][..., 0] += x_offset
            if y_offset:
                contours[0][..., 1] += y_offset

            # Get the geojson obj
            geo_obj = FileHandler.get_geo_obj(what=geo_format)
            poly = contours[0].squeeze().tolist()
            poly.append(poly[0])  # close the polygon
            geo_obj["geometry"]["coordinates"] = [poly]
            geo_obj["properties"]["classification"]["name"] = inst_type
            if soft_type is not None:
                geo_obj["properties"]["classification"][
                    "probabilities"
                ] = inst_type_soft

            geo_objs.append(geo_obj)

        return geo_objs

    @staticmethod
    def read_h5_patch(
        path: Union[Path, str],
        ix: int,
        return_im: bool = True,
        return_inst: bool = True,
        return_type: bool = True,
        return_sem: bool = False,
        return_name: bool = False,
        return_nitems: bool = False,
        return_all_names: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Read img & mask patches at index `ix` from a hdf5 db.

        Parameters
        ----------
            path : Path or str
                Path to the h5-db.
            ix : int
                Index for the hdf5 db-arrays.
            return_im : bool, default=True
                If True, returns an image. (If the db contains these.)
            return_inst : bool, default=True
                If True, returns a instance labelled mask. (If the db contains these.)
            return_type : bool, default=True
                If True, returns a type mask. (If the db contains these.)
            return_sem : bool, default=False
                If True, returns a semantic mask, (If the db contains these.)
            return_name : bool, default=False
                If True, returns a name for the patch, (If the db contains these.)
            return_nitems : bool, default=False
                If True, returns the number of items in the db.
            return_all_names : bool, default=False
                If True, returns all the names in the db.

        Returns
        -------
            Dict[str, np.ndarray]:
                A Dict of numpy matrices. Img shape: (H, W, 3), mask shapes: (H, W).
                keys of the dict are: "im", "inst", "type", "sem"

        Raises
        ------
            IOError: If a mask that does not exist in the db is being read.
        """
        if not _has_tb:
            raise ModuleNotFoundError(
                "`FileHandler.read_h5_patch` method requires pytables library. "
                "Install with `pip install tables`."
            )

        path = Path(path)
        with tb.open_file(path.as_posix(), "r") as h5:
            out = {}

            if return_im:
                try:
                    out["image"] = h5.root.imgs[ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain images. Try "
                        "setting `return_im=False`"
                    )

            if return_inst:
                try:
                    out["inst"] = h5.root.insts[ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain instance labelled masks. "
                        "Try setting `return_inst=False`"
                    )

            if return_type:
                try:
                    out["type"] = h5.root.types[ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain type masks. Try setting "
                        "`return_type = False` "
                    )

            if return_sem:
                try:
                    out["sem"] = h5.root.areas[ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain semantic masks. Try "
                        "setting `return_sem = False`"
                    )

            if return_name:
                try:
                    fn = h5.root.fnames[ix]
                    out["name"] = Path(fn.decode("UTF-8"))
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain patch names. Try "
                        "setting `return_name = False`"
                    )

            if return_nitems:
                try:
                    out["nitems"] = h5.root._v_attrs.n_items
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain attribute ``nitems. Try "
                        "setting `return_nitems = False`"
                    )

            if return_all_names:
                try:
                    names = h5.root.fnames[:]
                    out["names"] = [Path(n.decode("UTF-8")) for n in names]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain patch names. Try "
                        "setting `return_all_names = False`"
                    )

            return out

    @staticmethod
    def write_mat(
        fname: Union[str, Path],
        inst: np.ndarray,
        type: np.ndarray = None,
        sem: np.ndarray = None,
        compute_centorids: bool = False,
        compute_bboxes: bool = False,
        **kwargs,
    ) -> None:
        """
        Write multiple masks to .mat file.

        Keys always present in the file: "inst_map", "inst_type"

        Optional keys: "type_map", "sem_map", "inst_bbox", "inst_centroid"

        Parameters
        ----------
            fname : str or Path
                The file name of the .mat file.
            inst : np.ndarray
                Instance labelled mask. Shape: (H, W).
            type : np.ndarray
                Cell type labelled semantic segmentation mask. Shape: (H, W).
            sem : np.ndarray
                Tissue type labelled semantic segmentation mask. Shape: (H, W).
            compute_centroids : bool, optional
                Flag to tompute instance centorids.
            compute_bboxes : bool, optional
                Flag to tompute instance bboxes.
        """
        fname = Path(fname)
        if not fname.parent.exists():
            raise ValueError(
                f"The directory: {fname.parent.as_posix()} does not exist."
            )

        inst_map = fix_duplicates(inst)
        inst_types = get_inst_types(inst, type)

        res = {
            "inst_map": inst_map,
            "inst_type": inst_types,
        }

        if compute_centorids:
            centroids = get_inst_centroid(inst_map)
            res["inst_centroid"] = centroids

        if compute_bboxes:
            inst_ids = list(np.unique(inst_map)[1:])
            bboxes = np.array(
                [bounding_box(np.array(inst_map == id_, np.uint8)) for id_ in inst_ids]
            )
            res["inst_bbox"] = bboxes

        if type is not None:
            res["type_map"] = type

        if sem is not None:
            res["sem_map"] = sem

        sio.savemat(
            file_name=fname.with_suffix(".mat").as_posix(),
            mdict=res,
        )

    @staticmethod
    def write_gson(
        fname: Union[str, Path],
        inst: np.ndarray,
        type: np.ndarray = None,
        classes: Dict[str, int] = None,
        soft_type: np.ndarray = None,
        x_offset: int = 0,
        y_offset: int = 0,
        geo_format: str = "qupath",
    ) -> None:
        """Convert the instance labelled mask into geojson obj or write it .json file.

        Parameters
        ----------
            fname : str | Path, optional
                File name for the .json file.
            inst : np.ndarray
                Instance labelled mask. Shape: (H, W).
            type : np.ndarray, optional
                Cell type labelled semantic segmentation mask. Shape: (H, W). If None,
                the classes of the objects will be set to {background: 0, foreground: 1}
            soft_type : np.ndarray, default=None
                Softmax type mask. Shape: (C, H, W). C is the number of classes.
            classes : Dict[str, int], optional
                Class dict e.g. {"inflam":1, "epithelial":2, "connec":3}. Ignored if
                `type` is None.
            x_offset : int, default=0
                x-coordinate offset. (to set geojson to .mrxs wsi coordinates)
            y_offset : int, default=0
                y-coordinate offset. (to set geojson to .mrxs wsi coordinates)
            geo_format : str, default="qupath"
                The format for the geo object. "qupath" format allows the result file
                to be read with QuPath. "simple" format allows for geopandas etc.

        Raises
        ------
            ModuleNotFoundError: If geojson is not installed.
            ValueError: If `classes` is set to None when `type` is given.

        Returns
        -------
            Dict[str, Any]:
                A dictionary with geojson fields.
        """
        try:
            import geojson
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "To use the `FileHandler.mask2geojson`, geojson is required. "
                "Install with `pip install geojson`"
            )

        fname = Path(fname)
        if not fname.parent.exists():
            raise ValueError(
                f"The directory: {fname.parent.as_posix()} does not exist."
            )

        if type is None:
            type = inst > 0
            classes = {"background": 0, "foreground": 1}
        else:
            if classes is None:
                raise ValueError(
                    "`classes` cannot be None if `type` semgentation map is given."
                )

        geo_objs = FileHandler.get_gson(
            inst, type, classes, soft_type, x_offset, y_offset, geo_format
        )

        fname = fname.with_suffix(".json")

        with fname.open("w") as out:
            geojson.dump(geo_objs, out)

    @staticmethod
    def save_masks(
        fname: str,
        maps: Dict[str, np.ndarray],
        format: str = ".mat",
        json_format: str = "qupath",
        classes_type: Dict[str, str] = None,
        classes_sem: Dict[str, str] = None,
        offsets: bool = False,
        **kwargs,
    ) -> None:
        """Save model outputs to .mat or geojson .json file.

        NOTE: If .json format is used, two files are written if both inst_map and
        sem_map are given. The sem_map .json has a suffix '{}_areas.json' and
        the inst_map .json has suffix '{}_cells.json'

        Parameters
        ----------
            fname : str
                Name for the output-file.
            maps : Dict[str, np.ndarray]
                model output names mapped to model outputs.
                E.g. {"sem": np.ndarray, "type": np.ndarray, "inst": np.ndarray}
            format : str
                One of ".mat" or ".json"
            json_format : str, default="qupath"
                The geojson format. One of "qupath", "simple". Ignored if format is not
                ".json".
            classes_type : Dict[str, str], optional
                Cell type dictionary. e.g. {"inflam":1, "epithelial":2, "connec":3}.
                This is required only if `format` == `json`.
            classes_sem : Dict[str, str], otional
                Tissue type dictionary. e.g. {"tissue1":1, "tissue2":2, "tissue3":3}
                This is required only if `format` == `json`.
            offsets : bool, default=False
                If True, geojson coords are shifted by the offsets that are encoded in
                the filenames (e.g. "x-1000_y-4000.png"). Ignored if `format` != ".json"
        """
        fname = Path(fname)
        allowed = (".mat", ".json")
        if format not in allowed:
            raise ValueError(
                f"Illegal file-format. Got: {format}. Allowed formats: {allowed}"
            )

        if format == ".mat":
            FileHandler.write_mat(fname, **maps)
        elif format == ".json":
            offs = FileHandler.get_offset(fname.name) if offsets else {"x": 0, "y": 0}

            if "inst" in maps.keys():
                type_map = None
                if "type" in maps.keys():
                    type_map = maps["type"]

                # Create directory for the cell seg results
                save_dir = fname.parent / "cells"
                if not Path(save_dir).exists():
                    Path(save_dir).mkdir(parents=True, exist_ok=True)

                fn = save_dir / f"{fname.name}_cells"
                FileHandler.write_gson(
                    fname=fn,
                    inst=maps["inst"],
                    type=type_map,
                    classes=classes_type,
                    soft_type=maps["soft_type"] if "soft_type" in maps.keys() else None,
                    geo_format=json_format,
                    x_offset=offs["x"],
                    y_offset=offs["y"],
                )
            if "sem" in maps.keys():
                if classes_sem is None:
                    raise ValueError(
                        "When saving to .json `classes_sem` can't be None, "
                        "if the output masks contains tissue type predictions."
                    )

                # Create directory for the area seg results
                save_dir = fname.parent / "areas"
                if not Path(save_dir).exists():
                    Path(save_dir).mkdir(parents=True, exist_ok=True)

                fn = save_dir / f"{fname.name}_areas"

                FileHandler.write_gson(
                    fname=fn,
                    inst=label_semantic(maps["sem"]),
                    type=maps["sem"],
                    classes=classes_sem,
                    soft_type=maps["soft_sem"] if "soft_sem" in maps.keys() else None,
                    geo_format=json_format,
                    x_offset=offs["x"],
                    y_offset=offs["y"],
                )

    @staticmethod
    def save_masks_parallel(
        maps: List[Dict[str, np.ndarray]],
        fnames: List[str],
        format: str = ".mat",
        geo_format="qupath",
        classes_type: Dict[str, str] = None,
        classes_sem: Dict[str, str] = None,
        offsets: bool = False,
        pooltype: str = "thread",
        maptype: str = "amap",
        **kwargs,
    ) -> None:
        """Save the model output masks to a folder. (multi-threaded).

        Parameters
        ----------
            maps : List[Dict[str, np.ndarray]]
                The model output map dictionaries in a list.
            fnames : List[str]
                Name for the output-files. (In the same order with `maps`)
            format : str
                One of ".mat" or ".json"
            geo_format : str, default="qupath"
                The geojson format. One of "qupath", "simple". Ignored if format is not
                ".json".
            classes_type : Dict[str, str], optional
                Cell type dictionary. e.g. {"inflam":1, "epithelial":2, "connec":3}.
                This is required only if `format` == `json`.
            classes_sem : Dict[str, str], otional
                Tissue type dictionary. e.g. {"tissue1":1, "tissue2":2, "tissue3":3}
                This is required only if `format` == `json`.
            offsets : bool, default=False
                If True, geojson coords are shifted by the offsets that are encoded in
                the filenames (e.g. "x-1000_y-4000.png"). Ignored if `format` != ".json"
            pooltype : str, default="thread"
                The pathos pooltype. Allowed: ("process", "thread", "serial").
                Defaults to "thread". (Fastest in benchmarks.)
            maptype : str, default="amap"
                The map type of the pathos Pool object.
                Allowed: ("map", "amap", "imap", "uimap")
                Defaults to "amap". (Fastest in benchmarks).
        """
        func = partial(
            FileHandler._save_masks,
            format=format,
            geo_format=geo_format,
            classes_type=classes_type,
            classes_sem=classes_sem,
            offsets=offsets,
        )

        args = tuple(zip(fnames, maps))
        run_pool(func, args, ret=False, pooltype=pooltype, maptype=maptype)

    @staticmethod
    def _save_masks(
        args: Tuple[str, Dict[str, np.ndarray]],
        format: str,
        geo_format: str,
        classes_type: Dict[str, str],
        classes_sem: Dict[str, str],
        offsets: bool,
    ) -> None:
        """Unpacks the args for `save_mask` to enable multi-threading."""
        return FileHandler.save_masks(
            *args,
            format=format,
            geo_format=geo_format,
            classes_type=classes_type,
            classes_sem=classes_sem,
            offsets=offsets,
        )

    @staticmethod
    def get_split(string: str) -> List[str]:
        """Try splitting a coord-string with "-" and "_" on a string."""
        xy = string.split("-")
        if len(xy) > 1:
            return xy
        xy = string.split("_")
        if len(xy) > 1:
            return xy
        else:
            return list(filter(None, re.split(r"(\d+)", string)))

    @staticmethod
    def get_offset(fname: str) -> Dict[str, int]:
        """Get the offsets.

        I.e. If a filename contains x- and y- coordinates, return them.
        """
        coords = re.findall(r"([xy][ -_]\d+)", fname)
        offsets = {}
        for coord in coords:
            xy = FileHandler.get_split(coord)
            offsets[xy[0]] = int(xy[1])

        return offsets
