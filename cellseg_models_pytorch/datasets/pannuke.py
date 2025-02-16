import shutil
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from cellseg_models_pytorch.utils import (
    Downloader,
    FileHandler,
    H5Handler,
    fix_duplicates,
)

try:
    import tables as tb

    _has_tb = True
except ModuleNotFoundError:
    _has_tb = False

__all__ = ["Pannuke"]


class Pannuke:
    def __init__(
        self, save_dir: str, fold_split: Dict[str, str], verbose: bool = False
    ) -> None:
        """Pannuke dataset class."""
        self.save_dir = Path(save_dir)
        self.fold_split = fold_split
        self.verbose = verbose

        allowed_splits = ("train", "valid", "test")
        if not all([k in allowed_splits for k in fold_split.values()]):
            raise ValueError(
                f"`fold_split` values need to be in {allowed_splits}. "
                f"Got: {list(fold_split.values())}"
            )

        self.has_downloaded = self._check_if_downloaded()
        self.has_prepared_folders = self._check_if_folders_prepared()
        self.has_prepared_h5 = self._check_if_h5_prepared()

    @property
    def train_image_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("train", is_mask=False)

    @property
    def valid_image_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("valid", is_mask=False)

    @property
    def test_image_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("test", is_mask=False)

    @property
    def train_label_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("train", is_mask=True)

    @property
    def valid_label_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("valid", is_mask=True)

    @property
    def test_label_dir(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_dir("test", is_mask=True)

    @property
    def train_h5_file(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_h5("train")

    @property
    def valid_h5_file(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_h5("valid")

    @property
    def test_h5_file(self) -> Path:
        """Path to the train directory."""
        return self._get_fold_h5("test")

    @property
    def type_classes(self) -> Dict[int, str]:
        """Pannuke cell type classes."""
        return {
            0: "bg",
            1: "neoplastic",
            2: "inflammatory",
            3: "connective",
            4: "dead",
            5: "epithelial",
        }

    def download(self, root: str) -> None:
        """Download the pannuke dataset from online."""
        # create save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # init downloader
        downloader = Downloader(self.save_dir)
        for ix in [1, 2, 3]:
            url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{ix}.zip"
            downloader.download(url)
        FileHandler.extract_zips_in_folder(root, rm=True)

    def prepare_data(self, rm_orig: bool = False, to_h5: bool = False) -> None:
        """Prepare the pannuke datasets.

        1. Download pannuke folds from:
            "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/"
        2. Pre-process and split the images and masks into train, valid and test sets.

        Parameters:
            rm_orig (bool, default=False):
                After processing all the files, If True, removes the original
                un-processed files.
            to_h5 (bool, default=False):
                If True, saves the processed images and masks in one HDF5 file.
        """
        if not self.has_downloaded:
            if self.verbose:
                print(f"Downloading three Pannuke folds to {self.save_dir}")
            self.download(self.save_dir)

        if not self.has_prepared_folders:
            fold_paths = self._get_fold_paths(self.save_dir)
            for fold, phase in self.fold_split.items():
                save_im_dir = self.save_dir / phase / "images"
                save_mask_dir = self.save_dir / phase / "labels"

                self._prepare_data(
                    fold_paths, fold, phase, save_im_dir, save_mask_dir, h5path=None
                )
        if not self.has_prepared_h5 and to_h5:
            fold_paths = self._get_fold_paths(self.save_dir)
            for fold, phase in self.fold_split.items():
                h5path = self.save_dir / f"{phase}.h5"
                self._prepare_data(
                    fold_paths,
                    fold,
                    phase,
                    save_im_dir=None,
                    save_mask_dir=None,
                    h5path=h5path,
                )
        else:
            print(
                "Found pre-processed Pannuke data. If in need of a re-download, please empty the `save_dir` folder."
            )

        if rm_orig:
            for d in self.save_dir.iterdir():
                if "fold" in d.name.lower():
                    shutil.rmtree(d)

    def _prepare_data(
        self,
        fold_paths,
        fold,
        phase,
        save_im_dir: Path,
        save_mask_dir: Path,
        h5path: Path,
    ):
        # determine fold number
        if isinstance(fold, int):
            fold_ix = fold
        else:
            fold_ix = int(fold[-1])

        self._process_pannuke_fold(
            fold_paths, fold_ix, phase, save_im_dir, save_mask_dir, h5path
        )

    def _check_if_downloaded(self) -> bool:
        # check if the pannuke data has been downloaded
        if self.save_dir.exists() and self.save_dir.is_dir():
            folds_found = [
                d.name
                for d in self.save_dir.iterdir()
                if "fold" in d.name.lower() and d.is_dir()
            ]
            if len(folds_found) == 3:
                if self.verbose:
                    print(
                        f"Found all Pannuke folds {folds_found} inside {self.save_dir}."
                    )
                return True
        return False

    def _check_if_folders_prepared(self) -> True:
        # check if the pannuke data has been processed
        if self.save_dir.exists() and self.save_dir.is_dir():
            phases_found = [
                d.name
                for d in self.save_dir.iterdir()
                if d.name in ("train", "test", "valid") and d.is_dir()
            ]
            if phases_found:
                if self.verbose:
                    print(
                        f"Found processed Pannuke data saved in {phases_found} folders "
                        f"inside {self.save_dir}."
                    )
                return True
        return False

    def _check_if_h5_prepared(self) -> bool:
        # check if the pannuke data has been processed
        if self.save_dir.exists() and self.save_dir.is_dir():
            h5_found = [d.name for d in self.save_dir.iterdir() if d.suffix == ".h5"]
            if h5_found:
                if self.verbose:
                    print(
                        f"Found processed Pannuke data saved in {h5_found} hdf5 files "
                        f"inside {self.save_dir}."
                    )
                return True
        return False

    def _get_fold_paths(self, path: Path) -> Dict[str, Path]:
        """Get the paths to the .npy files in all of the fold folders."""
        return {
            f"{file.parts[-2]}_{file.name[:-4]}": file
            for dir1 in path.iterdir()
            if dir1.is_dir()
            for dir2 in dir1.iterdir()
            if dir2.is_dir()
            for dir3 in dir2.iterdir()
            if dir3.is_dir()
            for file in dir3.iterdir()
            if file.is_file() and file.suffix == ".npy"
        }

    def _process_pannuke_fold(
        self,
        fold_paths: Dict[str, Path],
        fold: int,
        phase: str,
        save_im_dir: Path = None,
        save_mask_dir: Path = None,
        h5path: Path = None,
    ) -> None:
        """Save the pannuke patches .mat files in 'train', 'valid' & 'test' folders."""
        if h5path is not None:
            if not _has_tb:
                raise ModuleNotFoundError(
                    "Please install `tables` to save the data in HDF5 format."
                )
            h5handler = H5Handler()
            patch_size = (256, 256)

            h5 = tb.open_file(h5path, "w")
            try:
                h5handler.init_img(h5, patch_size)
                h5handler.init_mask(h5, "inst", patch_size)
                h5handler.init_mask(h5, "type", patch_size)
                h5handler.init_meta_data(h5)
            except Exception as e:
                h5.close()
                raise e
        else:
            # Create directories for the files.
            Path(save_im_dir).mkdir(parents=True, exist_ok=True)
            Path(save_mask_dir).mkdir(parents=True, exist_ok=True)

        masks = np.load(fold_paths[f"fold{fold}_masks"]).astype("int32")
        imgs = np.load(fold_paths[f"fold{fold}_images"]).astype("uint8")
        types = np.load(fold_paths[f"fold{fold}_types"])

        with tqdm(total=len(types)) as pbar:
            pbar.set_description(f"fold{fold}/{phase}")
            for tissue_type in np.unique(types):
                imgs_by_type = imgs[types == tissue_type]
                masks_by_type = masks[types == tissue_type]
                for j in range(imgs_by_type.shape[0]):
                    im = imgs_by_type[j, ...]
                    temp_mask = masks_by_type[j, ...]
                    type_map = self._get_type_map(temp_mask)
                    inst_map = self._get_inst_map(temp_mask[..., 0:5])
                    name = f"{tissue_type}_fold{fold}_{j}"

                    if h5path is not None:
                        try:
                            h5handler.append_array(h5, im[None, ...], "image")
                            h5handler.append_array(h5, inst_map[None, ...], "inst")
                            h5handler.append_array(h5, type_map[None, ...], "type")
                            h5handler.append_meta_data(
                                h5, name, coords=(0, 0, 256, 256)
                            )
                        except Exception as e:
                            h5.close()
                            raise e
                    else:
                        fn_im = Path(save_im_dir / name).with_suffix(".png")
                        FileHandler.write_img(fn_im, im)

                        fn_mask = Path(save_mask_dir / name).with_suffix(".mat")
                        FileHandler.to_mat(
                            masks={
                                "inst": inst_map,
                                "type": type_map,
                            },
                            path=fn_mask,
                        )
                    pbar.update(1)

        if h5path is not None:
            h5.close()

    def _get_type_map(self, pannuke_mask: np.ndarray) -> np.ndarray:
        """Convert the pannuke mask to type map of shape (H, W)."""
        mask = pannuke_mask.copy()
        mask[mask > 0] = 1

        # init type_map and set the background channel
        # of the pannuke mask as the first channel
        type_map = np.zeros_like(mask)
        type_map[..., 0] = mask[..., -1]
        for i, j in enumerate(range(1, mask.shape[-1])):
            type_map[..., j] = mask[..., i]

        return np.argmax(type_map, axis=-1)

    def _get_inst_map(self, pannuke_mask: np.ndarray) -> np.ndarray:
        """Convert pannuke mask to inst_map of shape (H, W)."""
        mask = pannuke_mask.copy()

        inst_map = np.zeros(mask.shape[:2], dtype="int32")
        for i in range(mask.shape[-1]):
            insts = mask[..., i]
            inst_ids = np.unique(insts)[1:]
            for inst_id in inst_ids:
                inst = np.array(insts == inst_id, np.uint8)
                inst_map[inst > 0] += inst_id

        # fix duplicated instances
        inst_map = fix_duplicates(inst_map)
        return inst_map

    def _check_fold_exists(self, fold: str) -> bool:
        """Check if the fold exists in the save_dir."""
        for d in self.save_dir.iterdir():
            if d.is_dir():
                if fold in d.name:
                    return True
        return False

    def _check_file_exists(self, fold: str) -> bool:
        """Check if the fold exists in the save_dir."""
        for d in self.save_dir.iterdir():
            if d.is_file():
                if fold in d.name:
                    return True
        return False

    def _get_fold_dir(self, fold: str, is_mask: bool = False) -> Path:
        dir_name = "images"
        if is_mask:
            dir_name = "labels"

        if self._check_fold_exists(fold):
            return self.save_dir / f"{fold}/{dir_name}"
        else:
            raise ValueError(
                f"No '{fold}' directory found in {self.save_dir}. Expected a directory "
                f"named '{fold}/{dir_name}'. Run `prepare_data()` to create the data. "
                "and make sure that the fold exists in the `fold_split`."
            )

    def _get_fold_h5(self, fold: str) -> Path:
        if self._check_file_exists(fold):
            return self.save_dir / f"{fold}.h5"
        else:
            raise ValueError(
                f"No '{fold}' HDF5 file found in {self.save_dir}. Expected a file "
                f"named '{fold}.h5'. Run `prepare_data(to_h5=True)` to create the data. "
                "and make sure that the fold exists in the `fold_split`."
            )
