from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ...utils import FileHandler
from ._base_writer import BaseWriter

__all__ = ["SlidingWindowFolderWriter"]


class SlidingWindowFolderWriter(BaseWriter):
    def __init__(
        self,
        in_dir_im: str,
        in_dir_mask: str,
        save_dir_im: str,
        save_dir_mask: str,
        patch_size: Tuple[int, int],
        stride: int,
        transforms: Optional[List[str]] = None,
    ) -> None:
        """Write overlapping patches to a folder from image and .mat files.

        Image patches will be written as .png and mask patches as .mat files.

        Parameters
        ----------
            in_dir_im : str
                Path to the folder of images
            in_dir_mask : str
                Path to the folder of masks.
            save_dir_im : str
                Path to the folder where the new img patches will be saved.
            save_dir_mask : str
                Path to the folder where the new mask patches will be saved.
            patch_size : Tuple[int, int]
                Height and width of the extracted patches.
            stride : int
                Stride for the sliding window.
            transforms : List[str]:
                A list of transforms to apply to the images and masks. Allowed ones:
                "blur", "non_spatial", "non_rigid", "rigid", "hue_sat", "random_crop",
                "center_crop", "resize"

        Raises
        ------
            ValueError if issues with given paths or the files in those folders.

        Example
        -------
            >>> writer = SlidingWindowFolderWriter(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/img_patches/here/,
                    "/save/mask_patches/here/,
                    patch_size=(320, 320),
                    stride=160,
                    transforms=["rigid"]
                )
            >>> writer.write()

        """
        super().__init__(
            in_dir_im=in_dir_im,
            in_dir_mask=in_dir_mask,
            stride=stride,
            patch_size=patch_size,
            transforms=transforms,
        )
        self.save_dir_im = Path(save_dir_im)
        self.save_dir_mask = Path(save_dir_mask)

    def write(self) -> None:
        """Write patches to a folder."""
        with tqdm(
            zip(self.fnames_im, self.fnames_mask), total=len(self.fnames_im)
        ) as pbar:
            total_tiles = 0
            for fni, fnm in pbar:
                pbar.set_description("Extracting patches to folders..")
                tiles = self._get_tiles(fni, fnm)
                n_tiles = tiles["image"].shape[0]

                arg_list = []
                for i in range(n_tiles):
                    fn = fni.name[:-4]
                    dd = dict(
                        path_im=self.save_dir_im / f"{fn}_patch{i+1}.png",
                        path_mask=self.save_dir_mask / f"{fn}_patch{i+1}.mat",
                    )
                    for k, m in tiles.items():
                        dd[k] = m[i : i + 1].squeeze()
                    arg_list.append(dd)

                self._write_parallel(self._save2folder, arg_list)
                total_tiles += n_tiles
                pbar.set_postfix_str(f"# of extracted tiles {total_tiles}")

    def save2folder(
        self,
        path_im: str,
        path_mask: str,
        image: np.ndarray,
        inst_map: np.ndarray,
        type_map: np.ndarray,
        sem_map: np.ndarray,
    ) -> None:
        """Write image and corresponding masks to folder."""
        FileHandler.write_img(path_im, image)
        FileHandler.write_mask(path_mask, inst_map, type_map, sem_map)

    def _save2folder(self, kwargs) -> None:
        """Unpack the kwargs for `save2folder`."""
        return self.save2folder(**kwargs)
