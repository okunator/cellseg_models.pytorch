import warnings
import zipfile
from pathlib import Path
from typing import Dict, Union

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
    def extract_zips(path: Union[str, Path], rm: bool = False) -> None:
        """Extract files from all the .zip files inside a folder.

        Parameters
        ----------
            path : str or Path
                Path to a folder containing .zip files.
            rm :bool, default=False
                remove the .zip files after extraction.
        """
        for f in Path(path).iterdir():
            if f.is_file() and f.suffix == ".zip":
                with zipfile.ZipFile(f, "r") as z:
                    z.extractall(path)
                if rm:
                    f.unlink()
