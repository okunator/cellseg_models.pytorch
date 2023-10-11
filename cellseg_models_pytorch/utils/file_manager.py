import re
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np
import scipy.io as sio
from shapely.geometry import Polygon, mapping

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

try:
    import geopandas as gpd

    _has_gpd = True
except ModuleNotFoundError:
    _has_gpd = False


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
    def geo_obj(
        poly: Polygon,
        uid: Union[int, str],
        class_name: Optional[str] = None,
        class_probs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Return a __geo_interface__ feature object from a polygon.

        Parameters
        ----------
        poly : Polygon
            A shapely polygon to convert.
        uid : Union[int, str]
            The unique identifier of the polygon.
        class_name : str
            The name of the class.
        class_probs : [Dict[str, float]], default=None
            The probabilities of the classes.

        Returns
        -------
        Dict[str, Any]
            The geojson feature object.
        """
        feature = {
            "id": str(uid),
            "type": "Feature",
            "properties": {
                "id": str(uuid4()),
                "type": "Feature",
                "objectType": "annotation",
            },
            "geometry": mapping(poly),
        }

        if class_name is not None:
            feature["properties"]["classification"] = {}
            feature["properties"]["classification"]["name"] = class_name
            feature["properties"]["classification"]["color"] = None
            feature["properties"]["class_name"] = class_name

        if class_probs is not None:
            feature["properties"]["probabilities"] = class_probs

        return feature

    @staticmethod
    def get_gson(
        inst: np.ndarray,
        type: np.ndarray = None,
        classes: Dict[str, int] = None,
        soft_type: np.ndarray = None,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get the labels in geojson format.

        The return object implements the __geo_interface__ spec.

        Parameters
        ----------
            inst : np.ndarray
                Instance labelled mask. Shape: (H, W).
            type : np.ndarray, default=None
                Cell type labelled semantic segmentation mask. Shape: (H, W). If None
                the classes of the objects will be set to {background: 0, foreground: 1}
            classes : Dict[str, int], default=None
                Class dict e.g. {"inflam":1, "epithelial":2, "connec":3}. If None,
                the classes of the objects will be set to {background: 0, foreground: 1}
            soft_type : np.ndarray, default=None
                Softmax type mask. Shape: (C, H, W). C is the number of classes.
            x_offset : int, default=0
                x-coordinate offset. (to set geojson to .mrxs wsi coordinates)
            y_offset : int, default=0
                y-coordinate offset. (to set geojson to .mrxs wsi coordinates)

        Returns
        -------
            List[Dict[str, Any]]:
                A geojson dictionary of the instance labelled mask.
        """
        inst_map = fix_duplicates(inst)
        inst_list = list(np.unique(inst_map))
        if 0 in inst_list:
            inst_list.remove(0)

        if type is None or classes is None:
            type = inst > 0
            classes = {"background": 0, "foreground": 1}

        features = []
        for inst_id in inst_list:
            # set up the annotation geojson obj

            # Get cell instance and cell type
            inst = np.array(inst_map == inst_id, np.uint8)
            inst_type = type[inst_map == inst_id].astype("uint8")
            inst_type = np.unique(inst_type)[0]

            inst_type = [key for key in classes.keys() if classes[key] == inst_type][0]

            # type probabilities
            inst_type_soft = None
            if soft_type is not None:
                type_probs = soft_type[..., inst_map == inst_id].mean(axis=1)
                inst_type_soft = dict(zip(classes.keys(), type_probs))
                # convert to float for json serialization
                for key in inst_type_soft.keys():
                    inst_type_soft[key] = float(inst_type_soft[key])

            # get the cell contour coordinates
            contours = cv2.findContours(inst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            shell = contours[0]  # exterior
            holes = [cont for cont in contours[1:]]

            # got a line instead of a polygon
            if shell.shape[0] < 3:
                continue

            # shift coordinates based on the offsets
            if x_offset:
                shell[..., 0] += x_offset
                if holes:
                    for cont in holes:
                        cont[..., 0] += x_offset

            if y_offset:
                shell[..., 1] += y_offset
                if holes:
                    for cont in holes:
                        cont[..., 1] += y_offset

            # convert to list for shapely Polygon
            shell = shell.squeeze().tolist()
            if holes:
                holes = [cont.squeeze().tolist() for cont in holes]
            # shell.append(shell[0])  # close the polygon

            features.append(
                FileHandler.geo_obj(
                    poly=Polygon(shell=shell, holes=holes),
                    uid=inst_id,
                    class_name=inst_type,
                    class_probs=inst_type_soft,
                )
            )

        return features

    @staticmethod
    def to_gson(
        out_fn: Union[str, Path],
        features: List[Dict[str, Any]],
        format: str = ".feather",
        show_bbox: bool = True,
        silence_warnings: bool = True,
    ) -> None:
        """Write a geojson/feather/parquet file from a list of geojson features.

        Parameters
        ----------
        out_fn : Union[str, Path]
            The output filename.
        features : List[Dict[str, Any]]
            The list of geojson features.
        format : str, default="feather"
            The output format. One of ".feather", ".parquet", ".geojson".
        show_bbox : bool, default=True
            If True, the bbox is added to the geojson object.
        silence_warnings : bool, default=True
            If True, warnings are silenced.
        """
        out_fn = Path(out_fn)
        if format not in (".feather", ".parquet", ".geojson"):
            raise ValueError(
                f"Invalid format. Got: {format}. Allowed: feather, parquet, geojson"
            )

        if not _has_gpd:
            raise ModuleNotFoundError(
                "`FileHandler.to_gson` method requires the geopandas library. "
                "Install with `pip install geopandas`."
            )
        if features:
            geo = {
                "type": "FeatureCollection",
                "features": features,
            }

            gdf = gpd.GeoDataFrame.from_features(geo)
        else:
            # create empty gdf with col names to avoid errors
            gdf = gpd.GeoDataFrame(
                columns=[
                    "geometry",
                    "id",
                    "type",
                    "objectType",
                    "classification",
                    "class_name",
                ]
            )

        if not gdf.empty:
            gdf = gdf.set_geometry("geometry")

            # add the bbox
            if show_bbox:
                geo["bbox"] = tuple(gdf.total_bounds)
        else:
            if not silence_warnings:
                warnings.warn(f"The {out_fn.name} file is empty.")

        if format == ".feather":
            gdf.to_feather(out_fn.with_suffix(".feather"))
        elif format == ".parquet":
            gdf.to_parquet(out_fn.with_suffix(".parquet"))
        elif format == ".geojson":
            gdf.to_file(out_fn.with_suffix(".geojson"), driver="GeoJSON")

    @staticmethod
    def save_masks(
        fname: str,
        maps: Dict[str, np.ndarray],
        format: str = ".mat",
        classes_type: Dict[str, int] = None,
        classes_sem: Dict[str, int] = None,
        offsets: bool = False,
        **kwargs,
    ) -> None:
        """Save model outputs to .mat or geojson .json file.

        NOTE: If .json format is used, two files are written if both inst_map and
        sem_map are given. One for the inst_map and one for the sem_map. The files
        are saved in a folder named "cells" and "areas" respectively.'

        Parameters
        ----------
            fname : str
                Name for the output-file.
            maps : Dict[str, np.ndarray]
                model output names mapped to model outputs.
                E.g. {"sem": np.ndarray, "type": np.ndarray, "inst": np.ndarray}
            format : str
                One of ".mat" or ".json"
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
        fname = Path(fname).with_suffix(format)
        fn = fname
        allowed = (".mat", ".geojson", ".feather", ".parquet")
        if format not in allowed:
            raise ValueError(
                f"Illegal file-format. Got: {format}. Allowed formats: {allowed}"
            )

        if format == ".mat":
            FileHandler.write_mat(fname, **maps)
        elif format in (".geojson", ".feather", ".parquet"):
            x_off, y_off = (
                FileHandler.get_xy_coords(fname.name) if offsets else {"x": 0, "y": 0}
            )

            if "inst" in maps.keys():
                type_map = None
                if "type" in maps.keys():
                    type_map = maps["type"]

                # get the __geo_interface__ features
                geo_features = FileHandler.get_gson(
                    inst=maps["inst"],
                    type=type_map,
                    classes=classes_type,
                    soft_type=maps["soft_type"] if "soft_type" in maps.keys() else None,
                    x_offset=x_off,
                    y_offset=y_off,
                )

                # Create directory for the cell seg results if model outputs also
                # contain tissue type predictions
                if "sem" in maps.keys():
                    save_dir = fname.parent / "cells"
                    if not Path(save_dir).exists():
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fn = save_dir / f"{fname.stem}_cells"

                # save to file
                FileHandler.to_gson(
                    out_fn=fn,
                    features=geo_features,
                    format=format,
                )

            if "sem" in maps.keys():
                if classes_sem is None:
                    raise ValueError(
                        "When saving to .geojson `classes_sem` can't be None, "
                        "if the output masks contains tissue type predictions."
                    )

                geo_features = FileHandler.get_gson(
                    inst=label_semantic(maps["sem"]),
                    type=maps["sem"],
                    classes=classes_sem,
                    soft_type=maps["soft_sem"] if "soft_sem" in maps.keys() else None,
                    x_offset=x_off,
                    y_offset=y_off,
                )

                # Create directory for the tissue seg results if model outputs also
                # contain cell type predictions
                if "inst" in maps.keys():
                    save_dir = fname.parent / "areas"
                    if not Path(save_dir).exists():
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                    fn = save_dir / f"{fname.stem}_areas"

                # save to file
                FileHandler.to_gson(
                    out_fn=fn,
                    features=geo_features,
                    format=format,
                )

    @staticmethod
    def check_fn_format(fname: Union[Path, str]) -> None:
        """Check if the input filename correctly formatted coordinates.

        Parameters
        ----------
            fname : str
                The filename.

        Raises
        ------
            ValueError: If not all coordinates were found in filename.
            ValueError: If both x and y coordinates are not present in filename.
        """
        fn = Path(fname)
        allowed = (
            ".json",
            ".geojson",
            ".png",
            ".mat",
            ".h5",
            ".hdf5",
            ".feather",
            ".parquet",
        )
        if fn.suffix not in allowed:
            raise ValueError(
                f"Input file {fn} has wrong format. "
                "Expected '.png', '.json' or '.geojson'."
            )

        has_x = False
        has_y = False

        # get the x and y coordinates from the filename
        # NOTE: fname needs to contain x & y-coordinates in x_[coord1]_y_[coord2]-format
        # or x-[coord1]_y-[coord2]-format. The order of x and y can be any.
        xy_str: List[str] = re.findall(
            r"x\d+|y\d+|x_\d+|y_\d+|x-\d+|y-\d+", fn.as_posix()
        )

        try:
            for s in xy_str:
                if "x" in s:
                    has_x = True
                elif "y" in s:
                    has_y = True
        except IndexError:
            raise ValueError(
                "Not all coordinates were found in filename. "
                f"Filename has to be in 'x-[coord1]_y-[coord2]'-format, Got: {fn.name}"
            )

        if not has_x or not has_y:
            raise ValueError(
                "Both x and y coordinates have to be present in filename. "
                f"Got: {xy_str}. Filename has to be in 'x-[coord1]_y-[coord2]'-format."
            )

        return

    @staticmethod
    def get_xy_coords(fname: Union[Path, str]) -> Tuple[int, int]:
        """Get the x and y-coordinates from a filename.

        NOTE: The filename needs to contain x & y-coordinates in
            "x-[coord1]_y-[coord2]"-format

        Parameters
        ----------
            fname : str
                The filename. Has to contain x & y-coordinates

        Raises
        ------
            ValueError: If not the delimeter of x and y- coord is not '_' or '-'.

        Returns
        -------
            Tuple[int, int]: The x and y-coordinates in this order.
        """
        FileHandler.check_fn_format(fname)

        if isinstance(fname, Path):
            fname = fname.as_posix()

        xy_str: List[str] = re.findall(r"x\d+|y\d+|x_\d+|y_\d+|x-\d+|y-\d+", fname)
        xy: List[int] = [0, 0]
        for s in xy_str:
            if "x" in s:
                if "_" in s:
                    xy[0] = int(s.split("_")[1])
                elif "-" in s:
                    xy[0] = int(s.split("-")[1])
                elif "x" in s and "-" not in s and "_" not in s:
                    xy[0] = int(s.split("x")[1])
                else:
                    raise ValueError(
                        "The fname needs to contain x & y-coordinates in "
                        f"'x-[coord1]_y-[coord2]'-format. Got: {fname}"
                    )
            elif "y" in s:
                if "_" in s:
                    xy[1] = int(s.split("_")[1])
                elif "-" in s:
                    xy[1] = int(s.split("-")[1])
                elif "y" in s and "-" not in s and "_" not in s:
                    xy[1] = int(s.split("y")[1])
                else:
                    raise ValueError(
                        "The fname needs to contain x & y-coordinates in "
                        f"'x-[coord1]_y-[coord2]'-format. Got: {fname}"
                    )

        return xy[0], xy[1]

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
