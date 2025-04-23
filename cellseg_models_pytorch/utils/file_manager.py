import warnings
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import scipy.io as sio

from .mask_utils import bounding_box, get_inst_centroid, get_inst_types

try:
    import tables as tb
    from tables import File

    _has_tb = True
except ModuleNotFoundError:
    _has_tb = False

try:
    import geopandas as gpd

    _has_gpd = True
except ModuleNotFoundError:
    _has_gpd = False


__all__ = ["FileHandler", "H5Handler"]


class H5Handler:
    def __init__(self):
        if not _has_tb:
            raise ModuleNotFoundError(
                "`H5Handler` class requires pytables library. "
                "Install with `pip install tables`."
            )

    @staticmethod
    def init_mask(
        h5,
        name: str,
        patch_size: Tuple[int, int],
        chunk_size: int = 1,
        complevel: int = 5,
        complib: str = "blosc:lz4",
    ) -> None:
        """Initialize a mask array in the hdf5 file.

        Parameters:
            h5 (tb.file.File):
                The hdf5 file.
            name (str):
                The name of the mask array.
            patch_size (Tuple[int, int]):
                The size of the mask patches.
            chunk_size (int, default=1):
                The chunk size for the hdf5 array.
            complevel (int, default=5):
                The compression level.
            complib (str, default='blosc:lz4'):
                The compression library
        """
        h5.create_earray(
            where=h5.root,
            name=name,
            atom=tb.Int32Atom(),
            shape=np.append([0], patch_size),
            chunkshape=np.append([chunk_size], patch_size),
            filters=tb.Filters(complevel=complevel, complib=complib),
        )

    @staticmethod
    def init_img(
        h5,
        patch_size: Tuple[int, int],
        chunk_size: int = 1,
        complevel: int = 5,
        complib: str = "blosc:lz4",
    ) -> None:
        """Initialize an image array in the hdf5 file.

        Parameters:
            h5 (tb.file.File):
                The hdf5 file.
            patch_size (Tuple[int, int]):
                The size of the image patches.
            chunk_size (int, default=1):
                The chunk size for the hdf5 array.
            complevel (int, default=5):
                The compression level.
            complib (str, default='blosc:lz4'):
                The compression library.
        """
        h5.create_earray(
            where=h5.root,
            name="image",
            atom=tb.UInt8Atom(),
            shape=np.append([0], patch_size + (3,)),
            chunkshape=np.append([chunk_size], patch_size + (3,)),
            filters=tb.Filters(complevel=complevel, complib=complib),
        )

    @staticmethod
    def init_meta_data(
        h5,
        chunk_size: int = 1,
        complevel: int = 5,
        complib: str = "blosc:lz4",
    ) -> None:
        """Initialize meta data arrays in the hdf5 file.

        I.e. the filename and the coordinates of the patches. Coordinate format
        is (x0, y0, width, height).

        Parameters:
            h5 (tb.file.File):
                The hdf5 file.
            chunk_size (int, default=1):
                The chunk size for the hdf5 array.
            complevel (int, default=5):
                The compression level.
            complib (str, default='blosc:lz4'):
                The compression library.
        """
        h5.create_earray(
            where=h5.root,
            name="fname",
            atom=tb.StringAtom(itemsize=256),
            shape=(0,),
            filters=tb.Filters(complevel=complevel, complib=complib),
        )
        h5.create_earray(
            where=h5.root,
            name="coords",
            atom=tb.Int32Atom(),
            shape=(0, 4),
            chunkshape=(chunk_size, 4),
            filters=tb.Filters(complevel=complevel, complib=complib),
        )

    @staticmethod
    def init_h5(path: str, keys: Tuple[str, ...], patch_size: Tuple[int, int]):
        """Initialize a hdf5 file for saving masks.

        Parameters:
            path (str):
                The output path.
            keys (Tuple[str, ...]):
                The keys of the arrays to be saved.
            patch_size (Tuple[int, int]):
                The size of the mask patches.

        Returns:
            File:
                The hdf5 file.
        Raises:
            ModuleNotFoundError: If the tables library is not installed.
            ValueError: If invalid keys are given.
        """
        if not _has_tb:
            raise ModuleNotFoundError(
                "The tables lib is needed for saving in hdf5 format. "
                "Please install it using: `pip install tables`."
            )

        allowed_keys = ("inst", "type", "sem", "cyto_inst", "cyto_type")
        if not all(k in allowed_keys for k in keys):
            raise ValueError(
                f"Invalid keys. Allowed keys are {allowed_keys}, got {keys}"
            )

        try:
            h5 = tb.open_file(path, "w")

            if not h5.list_nodes("/"):
                for k in keys:
                    H5Handler.init_mask(h5, k, patch_size)
                H5Handler.init_meta_data(h5)
        except Exception as e:
            h5.close()
            raise e

        return h5

    @staticmethod
    def append_array(h5, array: np.ndarray, name: str) -> None:
        h5.root[name].append(array)

    @staticmethod
    def append_meta_data(h5, name: str, coords: Tuple[int, int]) -> None:
        h5.root["fname"].append(np.array([name], dtype=str))
        h5.root["coords"].append(np.array([coords], dtype=np.int32))


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
    def read_mat(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Read a .mat file.

        Parameters:
            path (str or Path):
                Path to the .mat file.

        Returns:
            Dict[str, np.ndarray]:
                A dictionary of numpy arrays.
        """
        path = Path(path)
        return sio.loadmat(path.as_posix())

    @staticmethod
    def read_h5(
        path: Union[Path, str], ix: int, keys: Tuple[str, ...]
    ) -> Dict[str, np.ndarray]:
        """Read img & mask patches at index `ix` from a hdf5 db.

        Parameters:
            path (Path or str):
                Path to the h5-db.
            ix (int):
                Index for the hdf5 db-arrays.
            keys (Tuple[str, ...]):
                Keys/Names of the arrays to be read.

        Returns:
            Dict[str, np.ndarray]:
                A Dict of numpy matrices. Img shape: (H, W, 3), mask shapes: (H, W).
                keys of the dict are: "im", "inst", "type", "cyto_inst", "cyto_type",
                "sem", "fname", "coords".
        Raises:
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
            for key in keys:
                try:
                    out[key] = h5.root[key][ix, ...]
                except Exception:
                    raise IOError(f"The HDF5 database does not contain '{key}' node.")

            return out

    @staticmethod
    def write_img(path: Union[str, Path], img: np.ndarray) -> None:
        """Write an image.

        Parameters:
            path (str or Path):
                Path to the image file.
            img (np.ndarray):
                The image to be written.

        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def gdf_to_file(
        gdf: gpd.GeoDataFrame, path: str, silence_warnings: bool = True
    ) -> None:
        """Write a GeoDataFrame to a file.

        Following formats are supported:
        - .feather
        - .parquet
        - .geojson

        Parameters:
            gdf (gpd.GeoDataFrame):
                The GeoDataFrame to be written.
            path (str or Path):
                The output filename.
            silence_warnings (bool, default=True):
                If True, warnings are silenced.

        Raises:
            ValueError: If an invalid format is given.

        Warnings:
            If the input GeoDataFrame is empty, a warning is raised.
        """
        if not _has_gpd:
            raise ModuleNotFoundError(
                "`FileHandler.gdf_to_file` method requires geopandas library. "
                "Install with `pip install geopandas`."
            )

        path = Path(path)
        suffix = path.suffix

        if gdf.empty:
            if not silence_warnings:
                warnings.warn("Input GeoDataFrame is empty. Nothing to write. Skip")
            return

        if suffix == ".feather":
            gdf.to_feather(path.with_suffix(".feather"))
        elif suffix == ".parquet":
            gdf.to_parquet(path.with_suffix(".parquet"))
        elif suffix == ".geojson":
            gdf.to_file(path.with_suffix(".geojson"), driver="GeoJSON")

    @staticmethod
    def to_mat(
        masks: Dict[str, np.ndarray],
        path: Union[str, Path],
        coords: Tuple[int, int, int, int] = None,
        compute_centroids: bool = False,
        compute_bboxes: bool = False,
        compute_type_array: bool = False,
    ) -> None:
        """Save masks to a .mat file.

        Note:
            Additonal computed arrays are saved if the corresponding flags are set:
            - centroids: The centroids of the instance masks. Shape (N, 2).
            - bbox: The bounding boxes of the instance masks. Shape (N, 4).
            - inst_type: The type array from the instance array. Shape (N, 1).

        Parameters:
            masks (Dict[str, np.ndarray]):
                The masks to be saved. E.g. {"inst": np.ndarray, "type": np.ndarray}.
            path (str or Path):
                The output path.
            coords (Tuple[int, int, int, int], default=None):
                The XYWH-coordinates of the image patch. (x0, y0, width, height).
            compute_centroids (bool, default=False):
                Compute the centroids of the instance masks.
            compute_bboxes (bool, default=False):
                Compute the bounding boxes of the instance masks.
            comute_type_array (bool, default=False):
                Compute the type array from the instance array. Shape (1, N).
        """
        fname = Path(path)
        res = {}

        if masks.get("inst", None) is not None:
            res["inst"] = masks["inst"]
            if compute_centroids:
                res["centroid"] = get_inst_centroid(masks["inst"])
            if compute_bboxes:
                inst_ids = list(np.unique(masks["inst"])[1:])
                res["bbox"] = np.array(
                    [
                        bounding_box(np.array(masks["inst"] == id_, np.uint8))
                        for id_ in inst_ids
                    ]
                )
            if compute_type_array and masks.get("type", None) is not None:
                res["inst_type"] = get_inst_types(masks["inst"], masks["type"])

        if masks.get("type", None) is not None:
            res["type"] = masks["type"]

        if masks.get("sem", None) is not None:
            res["sem"] = masks["sem"]

        if masks.get("cyto", None) is not None:
            res["cyto"] = masks["cyto"]

        if coords is not None:
            res["coords"] = np.array([coords])

        sio.savemat(
            file_name=fname.with_suffix(".mat").as_posix(),
            mdict=res,
        )

    @staticmethod
    def to_h5(
        masks: Dict[str, np.ndarray], h5, coords: Tuple[int, int, int, int]
    ) -> None:
        """Write masks to a hdf5 file.

        Parameters:
            masks (Dict[str, np.ndarray]):
                The masks to be written.
            h5 (tb.file.File):
                The hdf5 file.
            coords (Tuple[int, int, int, int]):
                The XYWH-coordinates of the image patch. (x0, y0, width, height).
        """
        try:
            if not h5.isopen:
                raise IOError(f"The hdf5 file {h5.filename} is not open. Can't write.")

            fname = Path(h5.filename).stem
            name = f"{fname}-x{coords[0]}-y{coords[1]}-w{coords[2]}-y{coords[3]}"
            h5handler = H5Handler()

            if masks.get("image", None) is not None:
                h5handler.append_array(h5, masks["image"][None, ...], "image")

            if masks.get("inst", None) is not None:
                h5handler.append_array(h5, masks["inst"][None, ...], "inst")

            if masks.get("type", None) is not None:
                h5handler.append_array(h5, masks["type"][None, ...], "type")

            if masks.get("cyto_inst", None) is not None:
                h5handler.append_array(h5, masks["cyto_inst"][None, ...], "cyto_inst")

            if masks.get("cyto_type", None) is not None:
                h5handler.append_array(h5, masks["cyto_type"][None, ...], "cyto_type")

            if masks.get("sem", None) is not None:
                h5handler.append_array(h5, masks["sem"][None, ...], "sem")

            h5handler.append_meta_data(h5, name, coords=coords)
        except Exception as e:
            h5.close()
            raise e

    @staticmethod
    def extract_zips_in_folder(path: Union[str, Path], rm: bool = False) -> None:
        """Extract files from all the .zip files inside a folder.

        Parameters:
            path (str or Path):
                Path to a folder containing .zip files.
            rm (bool, default=False):
                remove the .zip files after extraction.
        """
        for f in Path(path).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, "r") as z:
                    z.extractall(path)
                if rm:
                    f.unlink()
