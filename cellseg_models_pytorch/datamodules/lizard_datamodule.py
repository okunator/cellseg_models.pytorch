import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..datasets import SegmentationFolderDataset, SegmentationHDF5Dataset
    from ..datasets.dataset_writers.folder_writer import FolderWriter
    from ..datasets.dataset_writers.hdf5_writer import HDF5Writer
    from ._basemodule import BaseDataModule
    from .downloader import SimpleDownloader
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To use the LizardDataModule, requests, lightning, & albumentations "
        "libraries are needed. Install with "
        "`pip install requests lightning albumentations`"
    )


class LizardDataModule(BaseDataModule):
    def __init__(
        self,
        save_dir: str,
        fold_split: Dict[str, int],
        img_transforms: List[str],
        inst_transforms: List[str],
        dataset_type: str = "folder",
        patch_size: Tuple[int, int] = (256, 256),
        stride: int = 128,
        normalization: str = None,
        batch_size: int = 8,
        num_workers: int = 8,
        **kwargs,
    ) -> None:
        """Set up Lizard datamodule. Creates overlapping patches of the Lizard dataset.

        The patches will be saved in directories:
        - `{save_dir}/train/*`
        - `{save_dir}/test/*`
        - `{save_dir}/valid/*`

        References
        ----------
        Graham, S., Jahanifar, M., Azam, A., Nimir, M., Tsang, Y.W., Dodd, K., Hero, E.,
        Sahota, H., Tank, A., Benes, K., & others (2021). Lizard: A Large-Scale Dataset
        for Colonic Nuclear Instance Segmentation and Classification. In Proceedings of
        the IEEE/CVF International Conference on Computer Vision (pp. 684-693).

        Parameters
        ----------
            save_dir : str
                Path to directory where the lizard data will be saved.
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
            dataset_type : str, default="folder"
                The dataset type. One of "folder", "hdf5".
            patch_size : Tuple[int, int], default=(256, 256)
                The size of the patch extracted from the images.
            stride : int, default=128
                The stride of the sliding window patcher.
            normalization : str, optional
                Apply img normalization after all the transformations. One of "minmax",
                "norm", "percentile", None.
            batch_size : int, default=8
                Batch size for the dataloader.
            num_workers : int, default=8
                number of cpu cores/threads used in the dataloading process.

        Example
        -------
            >>> from pathlib import Path
            >>> from cellseg_models_pytorch.datamodules import LizardDataModule

            >>> fold_split = {"train": 1, "valid": 2, "test": 3}
            >>> save_dir = Path.home() / "lizard"
            >>> lizard_module = LizardDataModule(
                    save_dir=save_dir,
                    fold_split=fold_split,
                    inst_transforms=["dist", "stardist"],
                    img_transforms=["blur", "hue_sat"],
                    normalization="percentile",
                    dataset_type="hdf5",
                    patch_size=(320, 320),
                    stride=128
                )

            >>> # lizard_module.download(save_dir) # just the downloading
            >>> lizard_module.prepare_data(tiling=True) # downloading & processing
        """
        super().__init__(batch_size, num_workers)
        self.save_dir = Path(save_dir)
        self.fold_split = fold_split
        self.patch_size = patch_size
        self.stride = stride
        self.img_transforms = img_transforms
        self.inst_transforms = inst_transforms
        self.normalization = normalization
        self.kwargs = kwargs if kwargs is not None else {}

        if dataset_type not in ("folder", "hdf5"):
            raise ValueError(
                f"Illegal `dataset_type` arg. Got {dataset_type}. "
                f"Allowed: {('folder', 'hdf5')}"
            )

        self.dataset_type = dataset_type

    @property
    def type_classes(self) -> Dict[str, int]:
        """Lizard cell type classes."""
        return {
            "bg": 0,
            "neutrophil": 1,
            "epithelial": 2,
            "lymphocyte": 3,
            "plasma": 4,
            "eosinophil": 5,
            "connective": 6,
        }

    @staticmethod
    def download(root: str) -> None:
        """Download the lizard dataset from online."""
        for ix in [1, 2]:
            fn = f"lizard_images{ix}.zip"
            url = f"https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/{fn}"
            SimpleDownloader.download(url, root)

        url = "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/lizard_labels.zip"
        SimpleDownloader.download(url, root)
        LizardDataModule.extract_zips(root, rm=True)

    def prepare_data(self, rm_orig: bool = False, tiling: bool = True) -> None:
        """Prepare the lizard datasets.

        1. Download lizard folds from:
            "https://warwick.ac.uk/fac/cross_fac/tia/data/lizard/"
        2. split the images and masks into train, valid and test sets.
        3. Patch the images such that they can be used with the datasets/loaders.

        Parameters
        ----------
            rm_orig : bool, default=False
                After processing all the files, If True, removes the original
                un-processed files.
            tiling : bool, default=True
                Flag, whether to cut images into tiles. Can be set to False if you only
                want to download and split the data and then work it out on your own.
        """
        folders_found = [
            d.name
            for d in self.save_dir.iterdir()
            if d.name.lower() in ("lizard_images1", "lizard_images2", "lizard_labels")
            and d.is_dir()
        ]
        phases_found = [
            d.name
            for d in self.save_dir.iterdir()
            if d.name in ("train", "test", "valid") and d.is_dir()
        ]

        patches_found = []
        if phases_found:
            patches_found = [
                sub_d.name
                for d in self.save_dir.iterdir()
                if d.name in ("train", "test", "valid") and d.is_dir()
                for sub_d in d.iterdir()
                if sub_d.name
                in (
                    f"{d.name}_im_patches",
                    f"{d.name}_mask_patches",
                    f"{d.name}_patches",
                )
                and any(sub_d.iterdir())
            ]

        if len(folders_found) < 3 and not phases_found:
            print(
                "Found no data or an incomplete dataset. Downloading the whole thing..."
            )
            for d in self.save_dir.iterdir():
                shutil.rmtree(d)
            LizardDataModule.download(self.save_dir)
        else:
            print("Found all folds. Skip downloading.")

        if not phases_found:
            print("Splitting the files into train, valid, and test sets.")
            for phase, fold_ix in self.fold_split.items():
                img_dir1 = self.save_dir / "Lizard_Images1"
                img_dir2 = self.save_dir / "Lizard_Images2"
                label_dir = self.save_dir / "Lizard_Labels"
                save_im_dir = self.save_dir / phase / "images"
                save_mask_dir = self.save_dir / phase / "labels"

                self._split_to_fold(
                    img_dir1,
                    img_dir2,
                    label_dir,
                    save_im_dir,
                    save_mask_dir,
                    fold_ix,
                    not rm_orig,
                )
        else:
            print(
                "Found splitted Lizard data. "
                "If in need of a re-download, please empty the `save_dir` folder."
            )

        if rm_orig:
            for d in self.save_dir.iterdir():
                if "lizard" in d.name.lower() or "macosx" in d.name.lower():
                    shutil.rmtree(d)

        if tiling and not patches_found:
            print("Patch the data... This will take a while...")
            for phase in self.fold_split.keys():
                save_im_dir = self.save_dir / phase / "images"
                save_mask_dir = self.save_dir / phase / "labels"

                if self.dataset_type == "hdf5":
                    sdir = self.save_dir / phase / f"{phase}_patches"
                    sdir.mkdir(parents=True, exist_ok=True)
                    writer = HDF5Writer(
                        in_dir_im=save_im_dir,
                        in_dir_mask=save_mask_dir,
                        save_dir=sdir,
                        file_name=f"lizard_{phase}.h5",
                        patch_size=self.patch_size,
                        stride=self.stride,
                        transforms=["rigid"],
                    )
                else:
                    sdir_im = self.save_dir / phase / f"{phase}_im_patches"
                    sdir_mask = self.save_dir / phase / f"{phase}_mask_patches"
                    sdir_im.mkdir(parents=True, exist_ok=True)
                    sdir_mask.mkdir(parents=True, exist_ok=True)
                    writer = FolderWriter(
                        in_dir_im=save_im_dir,
                        in_dir_mask=save_mask_dir,
                        save_dir_im=sdir_im,
                        save_dir_mask=sdir_mask,
                        patch_size=self.patch_size,
                        stride=self.stride,
                        transforms=["rigid"],
                    )
                writer.write(tiling=True, pre_proc=self._process_label, msg=phase)
        else:
            print(
                "Found processed Lizard data. "
                "If in need of a re-process, please empty the `save_dir` folders."
            )

    def _get_path(self, phase: str, dstype: str, is_mask: bool = False) -> Path:
        if dstype == "hdf5":
            p = self.save_dir / phase / f"{phase}_patches" / f"lizard_{phase}.h5"
        else:
            dtype = "mask" if is_mask else "im"
            p = self.save_dir / phase / f"{phase}_{dtype}_patches"

        return p

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the train, valid, and test datasets."""
        if self.dataset_type == "hdf5":
            DS = SegmentationHDF5Dataset
        else:
            DS = SegmentationFolderDataset

        self.trainset = DS(
            path=self._get_path("train", self.dataset_type, is_mask=False),
            mask_path=self._get_path("train", self.dataset_type, is_mask=True),
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

        self.validset = DS(
            path=self._get_path("valid", self.dataset_type, is_mask=False),
            mask_path=self._get_path("valid", self.dataset_type, is_mask=True),
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

        self.testset = DS(
            path=self._get_path("test", self.dataset_type, is_mask=False),
            mask_path=self._get_path("test", self.dataset_type, is_mask=True),
            img_transforms=self.img_transforms,
            inst_transforms=self.inst_transforms,
            return_sem=False,
            normalization=self.normalization,
            **self.kwargs,
        )

    def _split_to_fold(
        self,
        img_dir1: Path,
        img_dir2: Path,
        label_dir: Path,
        save_im_dir: Path,
        save_mask_dir: Path,
        fold: int,
        copy: bool = True,
    ) -> None:
        """Move the downloaded data split into one of 'train', 'valid' or 'test' dir."""
        Path(save_im_dir).mkdir(parents=True, exist_ok=True)
        Path(save_mask_dir).mkdir(parents=True, exist_ok=True)
        info_path = label_dir / "info.csv"
        info = np.genfromtxt(info_path, dtype="str", delimiter=",", skip_header=True)
        info = info[info[:, -1] == str(fold)]
        for i in range(info.shape[0]):
            fn, _, _ = info[i]

            p1 = img_dir1 / f"{fn}.png"
            p2 = img_dir2 / f"{fn}.png"
            src_im = p1 if p1.exists() else p2
            src_mask = label_dir / "Labels" / f"{fn}.mat"

            if copy:
                shutil.copy(src_im, save_im_dir)
                shutil.copy(src_mask, save_mask_dir)
            else:
                src_im.rename(save_im_dir / src_im.name)
                src_mask.rename(save_mask_dir / src_mask.name)

    def _process_label(self, label: np.ndarray) -> None:
        """Process the labels.

        NOTE: this is done to match the schema that's used by the Dataset classes.
        """
        inst_map = label["inst_map"]
        classes = label["class"]
        nuclei_id = label["id"]

        type_map = np.zeros_like(inst_map)
        unique_values = np.unique(inst_map).tolist()[1:]  # remove 0
        nuclei_id = np.squeeze(nuclei_id).tolist()
        for value in unique_values:
            # Get the position of the corresponding value
            inst = np.copy(inst_map == value)
            idx = nuclei_id.index(value)

            class_ = classes[idx]
            type_map[inst > 0] = class_

        return {"inst_map": inst_map, "type_map": type_map}
