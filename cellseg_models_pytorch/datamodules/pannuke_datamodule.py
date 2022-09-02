import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..utils import FileHandler, fix_duplicates

try:
    from ..datasets import SegmentationFolderDataset
    from ._basemodule import BaseDataModule
    from .downloader import SimpleDownloader
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the PannukeDataModule, requests, pytorch-lightning, & albumentations "
        "libraries are needed. Install with "
        "`pip install requests pytorch-lightning albumentations`"
    )


class PannukeDataModule(BaseDataModule):
    def __init__(
        self,
        save_dir: str,
        fold_split: Dict[str, int],
        img_transforms: List[str],
        inst_transforms: List[str],
        normalization: str = None,
        batch_size: int = 8,
        num_workers: int = 8,
        **kwargs,
    ) -> None:
        """Set up pannuke datamodule..

        References
        ----------
        Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019)
        PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation
        and classification. In European Congress on Digital Pathology (pp. 11-19).

        Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A.,
        Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and
        Baselines. arXiv preprint arXiv:2003.10778.

        License: https://creativecommons.org/licenses/by-nc-sa/4.0/

        Parameters
        ----------
            save_dir : str
                Path to directory where the pannuke data will be saved.
            fold_split : Dict[str, int]
                Defines how the folds are split into train, valid, and test sets.
                E.g. {"train": 1, "valid": 2, "test": 3}
            img_transforms : List[str]
                A list containing all the transformations that are applied to the input
                images and corresponding masks. Allowed ones: "blur", "non_spatial",
                "non_rigid", "rigid", "hue_sat", "random_crop", "center_crop", "resize"
            inst_transforms : List[str]
                A list containg all the transformations that are applied to only the
                instance labelled masks. Allowed ones: "cellpose", "contour", "dist",
                "edgeweight", "hovernet", "omnipose", "smooth_dist", "binarize"
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            batch_size : int, default=8
                Batch size for the dataloader.
            num_workers : int, default=8
                number of cpu cores/threads used in the dataloading process.
        """
        super().__init__(batch_size, num_workers)
        allowed = ("train", "valid", "test")
        if not all([k in allowed for k in fold_split.keys()]):
            raise ValueError(
                f"`fold_split` keys need to be in {allowed}. "
                f"Got: {list(fold_split.keys())}"
            )

        self.save_dir = Path(save_dir)
        self.fold_split = fold_split
        self.img_transforms = img_transforms
        self.inst_transforms = inst_transforms
        self.normalization = normalization
        self.kwargs = kwargs if kwargs is not None else {}

    @property
    def type_classes(self) -> Dict[str, int]:
        """Pannuke cell type classes."""
        return {
            "bg": 0,
            "neoplastic": 1,
            "inflammatory": 2,
            "connective": 3,
            "dead": 4,
            "epithelial": 5,
        }

    @staticmethod
    def download(root: str) -> None:
        """Download the pannuke dataset from online."""
        for ix in [1, 2, 3]:
            url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_{ix}.zip"
            SimpleDownloader.download(url, root)
        PannukeDataModule.extract_zips(root, rm=True)

    def prepare_data(self) -> None:
        """Prepare the pannuke datasets.

        1. Download pannuke folds from:
            "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/"
        2. Pre-process and split the images and masks into train, valid and test sets.
        """
        folds_found = [
            d.name
            for d in self.save_dir.iterdir()
            if "fold" in d.name.lower() and d.is_dir()
        ]
        phases_found = [
            d.name
            for d in self.save_dir.iterdir()
            if d.name in ("train", "test", "valid") and d.is_dir()
        ]

        if len(folds_found) < 3 and not phases_found:
            print(f"Found {len(folds_found)} folds. " "Downloading all three...")
            PannukeDataModule.download(self.save_dir)
        else:
            print("Found all folds. Skip downloading.")

        if not phases_found:
            print("Processing files...")
            fold_paths = self._get_fold_paths(self.save_dir)

            for phase, fold_ix in self.fold_split.items():
                save_im_dir = self.save_dir / phase / "images"
                save_mask_dir = self.save_dir / phase / "labels"
                self._process_pannuke_fold(
                    fold_paths, save_im_dir, save_mask_dir, fold_ix, phase
                )
        else:
            print(
                "Found processed pannuke data. "
                "If in need of a re-download, please empty the `save_dir` folder."
            )

        for d in self.save_dir.iterdir():
            if "fold" in d.name.lower():
                shutil.rmtree(d)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the train, valid, and test datasets."""
        self.trainset = SegmentationFolderDataset(
            path=self.save_dir / "train" / "images",
            mask_path=self.save_dir / "train" / "labels",
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

        self.validset = SegmentationFolderDataset(
            path=self.save_dir / "valid" / "images",
            mask_path=self.save_dir / "valid" / "labels",
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

        self.testset = SegmentationFolderDataset(
            path=self.save_dir / "test" / "images",
            mask_path=self.save_dir / "test" / "labels",
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

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
        save_im_dir: Path,
        save_mask_dir: Path,
        fold: int,
        phase: str,
    ) -> None:
        """Save the pannuke patches .mat files in 'train' and 'test' folders."""
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
                    name = f"{tissue_type}_fold{fold}_{j}"
                    fn_im = Path(save_im_dir / name).with_suffix(".png")
                    FileHandler.write_img(fn_im, imgs_by_type[j, ...])

                    temp_mask = masks_by_type[j, ...]
                    type_map = self._get_type_map(temp_mask)
                    inst_map = self._get_inst_map(temp_mask[..., 0:5])

                    fn_mask = Path(save_mask_dir / name).with_suffix(".mat")
                    FileHandler.write_mask(fn_mask, inst_map, type_map)
                    pbar.update(1)

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
