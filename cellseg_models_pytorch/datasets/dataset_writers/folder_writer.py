from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ...utils import FileHandler
from ._base_writer import BaseWriter

__all__ = ["FolderWriter"]


class FolderWriter(BaseWriter):
    def __init__(
        self,
        in_dir_im: str,
        save_dir_im: str,
        in_dir_mask: Optional[str] = None,
        save_dir_mask: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        stride: Optional[int] = None,
        transforms: Optional[List[str]] = None,
    ) -> None:
        """Write overlapping patches to a folder from image and .mat files.

        Image patches will be written as .png and mask patches as .mat files.

        NOTE: Very ad-hoc and not tested so there exists a chance of failure...

        Parameters
        ----------
            in_dir_im : str
                Path to the folder of images
            save_dir_im : str
                Path to the folder where the new imgs will be saved.
            in_dir_mask : str, optional
                Path to the folder of masks.
            save_dir_mask : str, optional
                Path to the folder where the new masks will be saved.
            patch_size : Tuple[int, int], optional
                Height and width of the extracted patches.
            stride : int, optional
                Stride for the sliding window.
            transforms : List[str], optional
                A list of transforms to apply to the images and masks. Allowed ones:
                "blur", "non_spatial", "non_rigid", "rigid", "hue_sat", "random_crop",
                "center_crop", "resize"

        Raises
        ------
            ValueError if issues with given paths or the files in those folders.

        Example
        -------
            >>> # Patch and write 2 folder
            >>> writer = FolderWriter(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/img_patches/here/,
                    "/save/mask_patches/here/,
                    patch_size=(320, 320),
                    stride=160,
                    transforms=["rigid"]
                )
            >>> writer.write(tiling=True)

            >>> # Don't patch, just write 2 folder
            >>> writer = FolderWriter(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/img_patches/here/,
                    "/save/mask_patches/here/,
                    transforms=["rigid"]
                )
            >>> writer.write(tiling=False)

        """
        super().__init__(
            in_dir_im=in_dir_im,
            in_dir_mask=in_dir_mask,
            stride=stride,
            patch_size=patch_size,
            transforms=transforms,
        )
        self.save_dir_im = Path(save_dir_im)

        if save_dir_mask is not None:
            self.save_dir_mask = Path(save_dir_mask)

    def write(
        self, tiling: bool = False, pre_proc: Callable = None, msg: str = None
    ) -> None:
        """Write patches to a folder.

        Parameters
        ----------
            tiling : bool, default=False
                Apply tiling to the images before saving.
            pre_proc : Callable, optional
                An optional pre-processing function for the masks.
        """
        it = self.fnames_im
        if self.fnames_mask is not None:
            it = zip(self.fnames_im, self.fnames_mask)

        with tqdm(it, total=len(self.fnames_im)) as pbar:
            total_tiles = 0
            for fn in pbar:
                # get the img and masks filenames
                if self.fnames_mask is not None:
                    fn_im, fn_mask = fn
                else:
                    fn_im = fn
                    fn_mask = None

                msg = msg if msg is not None else ""
                pbar.set_description(f"Extracting {msg} patches to folders..")

                # optionally patch and process images and masks
                im, masks = self.get_array(
                    fn_im, fn_mask, tiling=tiling, pre_proc=pre_proc
                )

                n_tiles = im.shape[0] if tiling else 1
                arg_list = []
                for i in range(n_tiles):
                    kw = self._set_kwargs(im, fn_im, masks, fn_mask, tiling, i)
                    arg_list.append(kw)

                self._write_parallel(self._save2folder, arg_list)
                total_tiles += n_tiles
                pbar.set_postfix_str(f"# of extracted tiles {total_tiles}")

    def _set_kwargs(
        self,
        im: np.ndarray,
        img_path: Path,
        masks: Dict[str, np.ndarray] = None,
        mask_path: Path = None,
        tiling: bool = False,
        ix: int = None,
    ) -> Dict[str, Any]:
        """Define one set of arguments for saving function."""
        fn = img_path.with_suffix("").name

        fn_im_new = f"{fn}.png"
        fn_mask_new = None
        if mask_path is not None:
            fn_mask_new = f"{fn}.mat"

        if tiling:
            fn_im_new = f"{fn}_patch{ix + 1}.png"

            if mask_path is not None:
                fn_mask_new = f"{fn}_patch{ix + 1}.mat"

        im_save_path = self.save_dir_im / fn_im_new

        mask_save_path = None
        if mask_path is not None:
            mask_save_path = self.save_dir_mask / fn_mask_new

        kwarg = {"path_im": im_save_path, "path_mask": mask_save_path}

        if tiling:
            kwarg["image"] = im[ix : ix + 1].squeeze()

            if mask_path is not None:
                for k, m in masks.items():
                    kwarg[k] = m[ix : ix + 1].squeeze()
        else:
            kwarg["image"] = im
            if mask_path is not None:
                for k, m in masks.items():
                    kwarg[k] = m

        return kwarg

    def save2folder(
        self,
        path_im: str,
        image: np.ndarray,
        path_mask: str = None,
        inst_map: np.ndarray = None,
        type_map: np.ndarray = None,
        sem_map: np.ndarray = None,
    ) -> None:
        """Write image and corresponding masks to folder."""
        FileHandler.write_img(path_im, image)
        if path_mask is not None:
            FileHandler.write_mat(path_mask, inst_map, type_map, sem_map)

    def _save2folder(self, kwargs) -> None:
        """Unpack the kwargs for `save2folder`."""
        return self.save2folder(**kwargs)
