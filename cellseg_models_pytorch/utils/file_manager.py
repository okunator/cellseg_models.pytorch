import warnings
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import scipy.io as sio

from .mask_utils import bounding_box, get_inst_centroid, get_inst_types
from .vectorize import inst2gdf, sem2gdf

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

            if "image" in keys:
                try:
                    out["image"] = h5.root["image"][ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain 'image' node.")

            if "inst" in keys:
                try:
                    out["inst"] = h5.root["inst"][ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain 'inst' node.")

            if "type" in keys:
                try:
                    out["type"] = h5.root["type"][ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain 'type' node.")

            if "cyto_inst" in keys:
                try:
                    out["cyto_inst"] = h5.root["cyto_inst"][ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain 'cyto_inst' node."
                    )

            if "cyto_type" in keys:
                try:
                    out["cyto_type"] = h5.root["cyto_type"][ix, ...]
                except Exception:
                    raise IOError(
                        "The HDF5 database does not contain 'cyto_type' node."
                    )

            if "sem" in keys:
                try:
                    out["sem"] = h5.root["sem"][ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain 'sem' node.")

            if "fname" in keys:
                try:
                    fn = h5.root.fnames[ix]
                    out["fname"] = Path(fn.decode("UTF-8"))
                except Exception:
                    raise IOError("The HDF5 database does not contain 'fname' node.")

            if "coords" in keys:
                try:
                    out["coords"] = h5.root["coords"][ix, ...]
                except Exception:
                    raise IOError("The HDF5 database does not contain 'coords' node.")

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
    def to_gson(
        masks: Dict[str, np.ndarray],
        path: Union[str, Path],
        xoff: int = None,
        yoff: int = None,
        compute_centroids: bool = False,
        compute_bboxes: bool = False,
        class_dict_inst: Dict[int, str] = None,
        class_dict_sem: Dict[int, str] = None,
        class_dict_cyto: Dict[int, str] = None,
        use_subfolders: bool = True,
        silence_warnings: bool = True,
    ) -> None:
        """Write a geojson/feather/parquet files from a dict of model output masks.

        Note:
            Additonal computed arrays are saved if the corresponding flags are set:
            - centroid: The centroids of the instance masks. Shape (N, 2).
            - bbox: The bounding boxes of the instance masks. Shape (N, 4).

        Note:
            Each specific mask type is either embedded to the end of the filename or
            saved to it's corresponding subfolder if `use_subfolders` is set to True:
            - inst: instance masks
            - sem: semantic masks
            - cyto: cytoplasm masks

        Parameters:
            masks (Dict[str, np.ndarray]):
                The masks to be saved. E.g. {"inst": np.ndarray, "type": np.ndarray}.
            path (str or Path):
                The output filename.
            xoff (int, default=None):
                The x-offset for the masks.
            yoff (int, default=None):
                The y-offset for the masks.
            compute_centroids (bool, default=False):
                Compute the centroids of the instance masks.
            compute_bboxes (bool, default=False):
                Compute the bounding boxes of the instance masks.
            class_dict_inst (Dict[int, str], default=None):
                A dictionary mapping class indices to class names.
                E.g. {1: 'neoplastic', 2: 'immune'}.
            class_dict_sem (Dict[int, str], default=None):
                A dictionary mapping class indices to class names.
                E.g. {1: 'tumor', 2: 'stroma'}.
            class_dict_cyto (Dict[int, str], default=None):
                A dictionary mapping class indices to class names.
                E.g. {1: 'neoplastic_cyto', 2: 'connective_cyto'}.
            use_subfolders (bool, default=True):
                If True, saves the masks to their respective subfolders. I.e. subfolders
                called 'inst', 'sem', 'cyto'. If False, the mask type is embedded in the
                end of the filename.
            silence_warnings (bool, default=True):
                If True, warnings are silenced.
        """
        path = Path(path)

        if masks.get("inst", None) is not None:
            if use_subfolders:
                inst_subdir = path.parent / "inst"
                inst_subdir.mkdir(parents=True, exist_ok=True)
                inst_path = inst_subdir / path.name
            else:
                inst_path = path.parent / f"{path.stem}_inst{path.suffix}"

            if masks.get("type", None) is not None:
                inst_gdf = inst2gdf(
                    masks["inst"],
                    masks["type"],
                    xoff=xoff,
                    yoff=yoff,
                    class_dict=class_dict_inst,
                )
            else:
                inst_gdf = inst2gdf(masks["inst"], xoff=xoff, yoff=yoff)

            if compute_centroids:
                inst_gdf["centroid"] = inst_gdf["geometry"].centroid
            if compute_bboxes:
                inst_gdf["bbox"] = inst_gdf["geometry"].apply(lambda x: x.bounds)

            FileHandler.gdf_to_file(inst_gdf, inst_path, silence_warnings)

        if masks.get("sem", None) is not None:
            if use_subfolders:
                sem_subdir = path.parent / "sem"
                sem_subdir.mkdir(parents=True, exist_ok=True)
                sem_path = sem_subdir / path.name
            else:
                sem_path = path.parent / f"{path.stem}_sem{path.suffix}"

            sem_gdf = sem2gdf(
                masks["sem"],
                xoff=xoff,
                yoff=yoff,
                class_dict=class_dict_sem,
            )
            FileHandler.gdf_to_file(sem_gdf, sem_path, silence_warnings)

        if masks.get("cyto", None) is not None:
            if use_subfolders:
                cyto_subdir = path.parent / "cyto"
                cyto_subdir.mkdir(parents=True, exist_ok=True)
                cyto_path = cyto_subdir / path.name
            else:
                cyto_path = path.parent / f"{path.stem}_cyto{path.suffix}"

            if masks.get("cyto_type", None) is not None:
                cyto_gdf = inst2gdf(
                    masks["cyto"],
                    masks["cyto_type"],
                    xoff=xoff,
                    yoff=yoff,
                    class_dict=class_dict_cyto,
                )
            else:
                cyto_gdf = inst2gdf(
                    masks["cyto"],
                    xoff=xoff,
                    yoff=yoff,
                    class_dict=class_dict_cyto,
                )
            FileHandler.gdf_to_file(cyto_gdf, cyto_path, silence_warnings)

    @staticmethod
    def to_mat(
        masks: Dict[str, np.ndarray],
        path: Union[str, Path],
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

        sio.savemat(
            file_name=fname.with_suffix(".mat").as_posix(),
            mdict=res,
        )

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
