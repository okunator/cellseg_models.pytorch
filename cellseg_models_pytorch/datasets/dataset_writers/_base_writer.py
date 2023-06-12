from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pathos.multiprocessing import ThreadPool as Pool

from ...transforms.albu_transforms import IMG_TRANSFORMS, compose
from ...utils import FileHandler, fix_duplicates, get_patches, remap_label

__all__ = ["BaseWriter"]

IMG_SUFFIXES = (".jpeg", ".jpg", ".tif", ".tiff", ".png")
MASK_SUFFIXES = (".mat",)


class BaseWriter(ABC):
    def __init__(
        self,
        in_dir_im: str,
        in_dir_mask: str = None,
        patch_size: Tuple[int, int] = None,
        stride: int = None,
        transforms: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Init base class for sliding window data writers."""
        self.stride = stride

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = patch_size
        self.im_dir = Path(in_dir_im)

        # Imgs
        if not self.im_dir.exists():
            raise ValueError(f"folder: {self.im_dir} does not exist")

        if not self.im_dir.is_dir():
            raise ValueError(f"path: {self.im_dir} is not a folder")

        im_files = []
        for types in IMG_SUFFIXES:
            im_files.extend(self.im_dir.glob(f"*{types}"))
        self.fnames_im = sorted(im_files)

        # Masks
        self.mask_dir = in_dir_mask
        self.fnames_mask = None
        if in_dir_mask is not None:
            self.mask_dir = Path(in_dir_mask)

            if not self.mask_dir.exists():
                raise ValueError(f"folder: {self.mask_dir} does not exist")

            if not self.mask_dir.is_dir():
                raise ValueError(f"path: {self.mask_dir} is not a folder")

            mask_files = []
            for types in MASK_SUFFIXES:
                mask_files.extend(self.mask_dir.glob(f"*{types}"))
            self.fnames_mask = sorted(mask_files)

            if len(self.fnames_im) != len(self.fnames_mask):
                raise ValueError(
                    f"Found different number of files in {self.im_dir.as_posix()} and "
                    f"{self.mask_dir.as_posix()}."
                )

        # Transformations
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

    def get_array(
        self,
        img_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        tiling: Optional[bool] = False,
        pre_proc: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, Union[None, Dict[str, np.ndarray]]]:
        """Pipeline that (optionally) patches and transforms input images and masks.

        Parameters
        ----------
            img_path : str or Path
                Path to an image file.
            mask_path : str or Path, optional
                Path to a .mat mask file.
            tiling : bool, default=False, optional
                Flag, whether to do tiling on the images (and masks).
            pre_proc : Callable, optional
                A pre-processing function that can be used to pre-process given input
                masks before the pipeline.

        Raises
        ------
            ValueError if self.stride or self.patch_size are not set to integer values.

        Returns
        -------
            Tuple[np.ndarray, Union[None, Dict[str, np.ndarray]]]:
                The processed image & masks. If `mask_path=None`, returns no masks.
                Img shape w/o tiling: (H, W, C). Dtype: uint8.
                Img shape w/ tiling: (N, pH, pW, C). Dtype: uint8.
                Masks w/o tiling: Shapes: (H, W). Dtypes: int32.
                Masks w/ tiling: Shapes: (N, pH, pW). Dtypes: int32.
        """
        im, masks = self._read_files(img_path, mask_path, pre_proc)

        # do tiling first if flag set to True
        if tiling:
            if not isinstance(self.stride, int) and not isinstance(
                self.patch_size, int
            ):
                raise ValueError(
                    "`self.stride` and `self.patch_size` need to be integers. Got: "
                    f"self.stride={self.stride}, self.patch_size={self.patch_size}"
                )

            im, masks = self._get_tiles(im, masks)

            if masks is not None:
                if "inst_map" in masks.keys():
                    masks["inst_map"] = self._fix_instances_tiles(masks["inst_map"])

            if self.transforms is not None:
                im, masks = self._transform_tiles(im, masks)
        else:
            if masks is not None:
                if "inst_map" in masks.keys():
                    masks["inst_map"] = self._fix_instances_one(masks["inst_map"])

            if self.transforms is not None:
                im, masks = self._transform_one(im, masks)

        return im, masks

    def _fix_instances_one(self, inst_map: np.ndarray) -> np.ndarray:
        """Fix duplicate instances and remap instance labels."""
        return remap_label(fix_duplicates(inst_map))

    def _read_files(
        self,
        img_path: Union[str, Path],
        mask_path: Union[str, Path] = None,
        pre_proc: Callable = None,
    ) -> Tuple[np.ndarray, Union[None, Dict[str, np.ndarray]]]:
        """Read image and corresponding masks if there are such."""
        im = FileHandler.read_img(img_path)

        masks = None
        if mask_path is not None:
            masks = FileHandler.read_mat(mask_path, return_all=True)

            if pre_proc is not None:
                masks = pre_proc(masks)

            masks = {
                key: arr
                for key, arr in masks.items()
                if key in ("inst_map", "type_map", "sem_map")
            }

        return im, masks

    def _get_tiles(
        self,
        im: np.ndarray,
        masks: Union[Dict[str, np.ndarray], None] = None,
    ) -> Tuple[Dict[str, np.ndarray], Union[Dict[str, np.ndarray], None]]:
        """Do tiling on an image and corresponding masks if there are such."""
        im_tiles = get_patches(im, self.stride, self.patch_size)[0]

        # Tile masks if there are such.
        mask_tiles = None
        if masks is not None:
            mask_tiles = {}
            inst = None
            types = None
            sem = None
            if "inst_map" in masks.keys():
                inst = masks["inst_map"]
            if "type_map" in masks.keys():
                types = masks["type_map"]
            if "sem_map" in masks.keys():
                sem = masks["sem_map"]

            if inst is not None:
                mask_tiles["inst_map"] = get_patches(
                    inst, self.stride, self.patch_size
                )[0]
            if types is not None:
                mask_tiles["type_map"] = get_patches(
                    types, self.stride, self.patch_size
                )[0]
            if sem is not None:
                mask_tiles["sem_map"] = get_patches(sem, self.stride, self.patch_size)[
                    0
                ]

        return im_tiles, mask_tiles

    def _transform_one(
        self, im: np.ndarray, masks: Dict[str, np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[Dict[str, np.ndarray], None]]:
        """Transform an image and corresponding mask if there is one."""
        if masks is not None:
            mask_names = [name for name in masks.keys()]
            masks = [mask for mask in masks.values()]
            out = self.transforms(image=im, masks=masks)
            masks = {n: mask for n, mask in zip(mask_names, out["masks"])}
        else:
            out = self.transforms(image=im)

        im = out["image"]

        return im, masks

    def _transform_tiles(
        self,
        im_tiles: np.ndarray,
        mask_tiles: Union[Dict[str, np.ndarray], None] = None,
    ) -> Tuple[np.ndarray, Union[Dict[str, np.ndarray], None]]:
        """Apply transformations to the tiles one by one."""
        n_tiles = im_tiles.shape[0]
        out_im_tiles = []

        out_mask_tiles = None
        if mask_tiles is not None:
            mask_names = [key for key in mask_tiles.keys()]
            out_mask_tiles = {k: [] for k in mask_tiles.keys()}

        for i in range(n_tiles):
            # Get one img tile
            im = im_tiles[i]

            # Get one set of mask tiles
            masks = None
            if mask_tiles is not None:
                masks = {n: mask_tiles[n][i] for n in mask_names}

            # transform imgs & masks
            im_tr, masks_tr = self._transform_one(im, masks)

            out_im_tiles.append(im_tr)
            if mask_tiles is not None:
                for mask_name in mask_names:
                    out_mask_tiles[mask_name].append(masks_tr[mask_name])

        # convert list of 2D-arrays to np.ndarray
        out_im_tiles = np.array(out_im_tiles)
        if mask_tiles is not None:
            for k, arr in out_mask_tiles.items():
                out_mask_tiles[k] = np.array(arr)

        return out_im_tiles, out_mask_tiles

    def _fix_instances_tiles(self, patches_inst: np.ndarray) -> np.ndarray:
        """Fix repeated labels and remap them in a patched instance labelled mask."""
        insts = []

        for i in range(patches_inst.shape[0]):
            insts.append(self._fix_instances_one(patches_inst[i]))

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
