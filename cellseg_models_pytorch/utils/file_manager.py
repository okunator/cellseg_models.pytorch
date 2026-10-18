import warnings
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import scipy.io as sio

from .mask_utils import bounding_box, get_inst_centroid, get_inst_types

try:
    import tables as tb
    from tables import File as File  # Keep the existing optional import alias.

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

        Args:
            h5 (tb.file.File): The hdf5 file.
            name: The name of the mask array.
            patch_size: The size of the mask patches.
            chunk_size: The chunk size for the hdf5 array.
                Defaults to 1.
            complevel: The compression level.
                Defaults to 5.
            complib: The compression library
                Defaults to 'blosc:lz4'.
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

        Args:
            h5 (tb.file.File): The hdf5 file.
            patch_size: The size of the image patches.
            chunk_size: The chunk size for the hdf5 array.
                Defaults to 1.
            complevel: The compression level.
                Defaults to 5.
            complib: The compression library.
                Defaults to 'blosc:lz4'.
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

        Args:
            h5 (tb.file.File): The hdf5 file.
            chunk_size: The chunk size for the hdf5 array.
                Defaults to 1.
            complevel: The compression level.
                Defaults to 5.
            complib: The compression library.
                Defaults to 'blosc:lz4'.
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

        Args:
            path: The output path.
            keys: The keys of the arrays to be saved.
            patch_size: The size of the mask patches.

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
    """Read RGB images, label arrays and selected vector output formats.

    Methods are static; call them on FileHandler without constructing an instance.
    They do not create output directories or guarantee exclusive file creation.
    """

    @staticmethod
    def read_img(path: Union[str, Path]) -> np.ndarray:
        """Read an image as RGB pixel values.

        Args:
            path: Image file readable by OpenCV.

        Returns:
            NumPy array (height, width, 3) in RGB channel order, normally uint8
            with values 0–255. OpenCV's default colour read discards alpha.
            No model normalization is applied. Decode failures raise OpenCV errors.
        """
        path = Path(path)
        return cv2.cvtColor(cv2.imread(path.as_posix()), cv2.COLOR_BGR2RGB)

    @staticmethod
    def read_mat(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Read arrays and metadata from a MATLAB file.

        Args:
            path: MATLAB file supported by scipy.io.loadmat; HDF5-based MATLAB v7.3
                files are not supported by this reader.

        Returns:
            Dictionary from scipy.io.loadmat, including stored arrays and metadata
            keys such as __header__, __version__ and __globals__. Shapes and dtypes
            are retained. Select your label keys explicitly, for example inst/type;
            class mappings are not inferred or remapped.
        """
        path = Path(path)
        return sio.loadmat(path.as_posix())

    @staticmethod
    def read_h5(
        path: Union[Path, str], ix: int, keys: Tuple[str, ...]
    ) -> Dict[str, np.ndarray]:
        """Read one sample from selected root-level HDF5 arrays.

        Args:
            path: Existing HDF5 file readable by PyTables.
            ix: Index on each selected array's first axis.
            keys: Root node names to read. Each must support node[ix, ...]; scalar
                metadata nodes are not sample arrays.

        Returns:
            Dictionary with exactly the requested keys and their NumPy sample values.
            Image nodes commonly contain HWC images and masks HW arrays, but this
            reader preserves stored shapes, dtypes, values and key names.

        Raises:
            ModuleNotFoundError: PyTables is unavailable.
            IOError: A requested node cannot be read. This wrapper reports missing
                nodes, invalid indices and other per-node reader failures this way.
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
        """Write an RGB image using OpenCV.

        Args:
            path: Destination path; its suffix selects the codec. The parent folder
                must already exist and an existing file may be overwritten.
            img: NumPy RGB image (height, width, 3), normally uint8 with values 0–255.
                Convert model-normalized floats to an appropriate image range first.

        Returns:
            None. This wrapper does not check OpenCV's success flag, so a failed
            write is not always reported as an exception. Verify output creation
            when the surrounding workflow needs a confirmed saved file.
        """
        path = Path(path)
        cv2.imwrite(path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def gdf_to_file(
        gdf: gpd.GeoDataFrame, path: str, silence_warnings: bool = True
    ) -> None:
        """Save a nonempty GeoDataFrame in a supported vector format.

        Args:
            gdf: GeoDataFrame containing geometries and their attributes. Coordinate
                units and coordinate reference system are preserved as supplied.
            path: Destination ending in .parquet, .feather or .geojson. Parent folders
                must exist; existing-file behavior follows the format writer.
            silence_warnings: Suppress the warning when an empty frame is skipped.

        Returns:
            None. Empty frames and unsupported suffixes produce no file. The method
            does not validate unsupported suffixes or signal this skip through a result.

        Raises:
            ModuleNotFoundError: GeoPandas is unavailable. Format-writer errors propagate.
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
        """Save selected segmentation masks in a MATLAB file.

        Args:
            masks: Dictionary of NumPy label arrays. Recognized keys are inst, type,
                sem and cyto; unrecognized keys, including cyto_inst/cyto_type, are
                ignored. Masks normally have HW axes. Instance 0 is background.
            path: Output filename; the suffix is replaced by .mat. Parent folders
                must exist and an existing file may be overwritten.
            coords: Optional (x, y, width, height) metadata, saved under coords.
                It does not shift mask pixels or derived measurements.
            compute_centroids: Add centroid as (x, y) pixel coordinates, in sorted
                instance-ID order. Assumes zero background is present.
            compute_bboxes: Add bbox from the inst mask. Assumes a background label
                is present, as the first sorted label is skipped.
            compute_type_array: Add inst_type when inst and type are provided.
                Uses the smallest class ID within each instance, not a majority vote.

        Returns:
            None. Saves recognized masks plus requested derived arrays. An input
            without recognized masks can still write a metadata-only or empty file.
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

        Args:
            masks: The masks to be written.
            h5 (tb.file.File): The hdf5 file.
            coords:
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
