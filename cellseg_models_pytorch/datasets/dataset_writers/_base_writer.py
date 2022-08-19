from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pathos.multiprocessing import ThreadPool as Pool

from ...transforms import IMG_TRANSFORMS, compose
from ...utils import FileHandler, TilerStitcher, fix_duplicates

__all__ = ["BaseWriter"]

IMG_SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")
MASK_SUFFIXES = (".mat",)


class BaseWriter(ABC):
    def __init__(
        self,
        in_dir_im: str,
        in_dir_mask: str,
        patch_size: Tuple[int, int],
        stride: int,
        transforms: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Init base class for sliding window data writers."""
        self.im_dir = Path(in_dir_im)
        self.mask_dir = Path(in_dir_mask)
        self.stride = stride
        self.patch_size = patch_size

        if not self.im_dir.exists():
            raise ValueError(f"folder: {self.im_dir} does not exist")

        if not self.im_dir.is_dir():
            raise ValueError(f"path: {self.im_dir} is not a folder")

        if not all([f.suffix in IMG_SUFFIXES for f in self.im_dir.iterdir()]):
            raise ValueError(
                f"files formats in given folder need to be in {IMG_SUFFIXES}"
            )

        if not self.mask_dir.exists():
            raise ValueError(f"folder: {self.mask_dir} does not exist")

        if not self.mask_dir.is_dir():
            raise ValueError(f"path: {self.mask_dir} is not a folder")

        if not all([f.suffix in MASK_SUFFIXES for f in self.mask_dir.iterdir()]):
            raise ValueError(
                f"files formats in given folder need to be in {MASK_SUFFIXES}"
            )

        self.fnames_im = sorted(self.im_dir.glob("*"))
        self.fnames_mask = sorted(self.mask_dir.glob("*"))
        if len(self.fnames_im) != len(self.fnames_mask):
            raise ValueError(
                f"Found different number of files in {self.im_dir.as_posix()} and "
                f"{self.mask_dir.as_posix()}."
            )

        self.transforms = None
        if transforms is not None:
            allowed = list(IMG_TRANSFORMS.keys())
            if not all([tr in allowed for tr in transforms]):
                raise ValueError(
                    f"Wrong transformation. Got: {transforms}. Allowed: {allowed}."
                )

            self.transforms = compose(
                [IMG_TRANSFORMS[tr](**kwargs) for tr in transforms]
            )

    @abstractmethod
    def write(self):
        """Patch images and mask to and write them to disk."""
        raise NotImplementedError

    def _get_tiles(
        self, img_path: Union[str, Path], mask_path: Union[str, Path]
    ) -> Dict[str, np.ndarray]:
        """Read one image and corresponding masks and do tiling on them."""
        im = FileHandler.read_img(img_path)
        masks = FileHandler.read_mask(mask_path, return_all=True)

        inst = None
        types = None
        sem = None
        if "inst_map" in masks.keys():
            inst = masks["inst_map"]
        if "type_map" in masks.keys():
            types = masks["type_map"]
        if "sem_map" in masks.keys():
            sem = masks["sem_map"]

        im_tiler = TilerStitcher(
            im_shape=im.shape, patch_shape=self.patch_size + (3,), stride=self.stride
        )

        mask_tiler = TilerStitcher(
            im_shape=inst.shape, patch_shape=self.patch_size + (1,), stride=self.stride
        )

        tiles = {}
        tiles["image"] = im_tiler.patch(im)
        if inst is not None:
            tiles["inst_map"] = self._fix_duplicates(mask_tiler.patch(inst).squeeze())
        if types is not None:
            tiles["type_map"] = mask_tiler.patch(types).squeeze()
        if sem is not None:
            tiles["sem_map"] = mask_tiler.patch(sem).squeeze()

        if self.transforms is not None:
            tiles = self._transform(tiles)

        return tiles

    def _transform(self, tiles: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply transformations to the tiles one by one."""
        n_tiles = tiles["image"].shape[0]
        masks = [arr for key, arr in tiles.items() if key != "image"]
        mask_names = [key for key in tiles.keys() if key != "image"]

        out_tiles = {k: [] for k in tiles.keys()}
        for i in range(n_tiles):
            m = [mask[i] for mask in masks]
            out = self.transforms(image=tiles["image"][i], masks=m)
            out_tiles["image"].append(out["image"])

            for j, mname in enumerate(mask_names):
                out_tiles[mname].append(out["masks"][j])

        for k, mask in out_tiles.items():
            out_tiles[k] = np.array(mask)

        return out_tiles

    def _fix_duplicates(self, patches_inst: np.ndarray) -> np.ndarray:
        """Fix repeatded labels in a patched instance labelled mask."""
        insts = []

        for i in range(patches_inst.shape[0]):
            insts.append(fix_duplicates(patches_inst[i]))

        insts = np.array(insts)

        return insts.astype("int32")

    def _write_parallel(self, write_func: Callable, arg_list: List) -> None:
        """Write patches in parallel."""
        with Pool() as pool:
            it = pool.imap(write_func, arg_list)
            for _ in it:
                pass

    def _write_sequential(self, write_func: Callable, arg_list: List) -> None:
        """Write patches sequentially."""
        for kwargs in arg_list:
            write_func(**kwargs)
