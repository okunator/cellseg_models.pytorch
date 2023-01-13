from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple

import numpy as np
import tables as tb
from tqdm import tqdm

from ...utils import FileHandler
from ._base_writer import BaseWriter

__all__ = ["HDF5Writer"]


class HDF5Writer(BaseWriter):
    def __init__(
        self,
        in_dir_im: str,
        save_dir: str,
        file_name: str,
        in_dir_mask: Optional[str] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        stride: Optional[int] = None,
        transforms: Optional[List[str]] = None,
        chunk_size: int = 1,
        complevel: int = 5,
        complib: str = "blosc:lz4",
    ) -> None:
        """Write overlapping patches to a hdf5 database.

        NOTE: Very ad-hoc and not tested so there exists a chance of failure...

        Parameters
        ----------
            in_dir_im : str
                Path to the folder of images
            in_dir_mask : str
                Path to the folder of masks.
            save_dir : str
                Path to the folder where the db will be saved.
            file_name : str
                Name of the h5 db-file.
            patch_size : Tuple[int, int]
                Height and width of the extracted patches.
            stride : int
                Stride for the sliding window.
            transforms : List[str]:
                A list of transforms to apply to the images and masks. Allowed ones:
                "blur", "non_spatial", "non_rigid", "rigid", "hue_sat", "random_crop",
                "center_crop", "resize"
            chunk_size : int, default=1
                The chunk size of the h5 arrays.
            complevel : int, default=5
                Specifies a compression level for data. The allowed range is 0-9.
                A value of 0 (the default) disables compression. (PyTables).
            complib : str, default="blosc:lz4"
                Specifies the compression library to be used. 'zlib', 'lzo', 'bzip2' and
                'blosc' are supported. Additional compressors for Blosc: 'blosc:blosclz'
                'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy', 'blosc:zlib' & 'blosc:zstd'.
                (PyTables).

        Raises
        ------
            ValueError if issues with given paths or the files in those folders.

        Example
        -------
            >>> # Patch and write to hdf5 db.
            >>> writer = HDF5Writer(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/db/here/,
                    "my_h5_dataset.h5",
                    patch_size=(320, 320),
                    stride=160,
                    transforms=["rigid"]
                )
            >>> writer.write(tiling=True)

            >>> # Don't patch, just write to hdf5 db.
            >>> writer = HDF5Writer(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/db/here/,
                    "my_h5_dataset.h5",
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

        self.chunk_size = chunk_size
        self.complevel = complevel
        self.complib = complib
        self.save_dir = Path(save_dir)
        self.fname = self.save_dir / file_name
        if self.fname.suffix:
            if self.fname.suffix not in (".h5", ".hdf5"):
                raise ValueError(
                    "`file_name` has to have a '.h5' or '.hdf5' suffix."
                    f"Got: {self.fname.suffix}"
                )
        else:
            self.fname = self.fname.with_suffix(".h5")

    def write(
        self,
        tiling: bool = False,
        pre_proc: Callable = None,
        msg: str = None,
    ) -> None:
        """Write patches to a hdf5 db.

        Parameters
        ----------
            tiling : bool, default=False
                Apply tiling to the images before saving.
            pre_proc : Callable, optional
                An optional pre-processing function for the masks.
        """
        h5 = tb.open_file(self.fname.as_posix(), mode="w")

        if self.patch_size is not None and tiling:
            psize = self.patch_size
        else:
            psize = FileHandler.read_img(self.fnames_im[0]).shape[:2]

        h5.create_earray(
            where=h5.root,
            name="imgs",
            atom=tb.UInt8Atom(),
            shape=np.append([0], psize + (3,)),
            chunkshape=np.append([self.chunk_size], psize + (3,)),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )
        h5.create_earray(
            where=h5.root,
            name="fnames",
            atom=tb.StringAtom(itemsize=256),
            shape=(0,),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )

        if self.fnames_mask is not None:
            h5.create_earray(
                where=h5.root,
                name="insts",
                atom=tb.Int32Atom(),
                shape=np.append([0], psize),
                chunkshape=np.append([self.chunk_size], psize),
                filters=tb.Filters(complevel=self.complevel, complib=self.complib),
            )

            h5.create_earray(
                where=h5.root,
                name="types",
                atom=tb.Int32Atom(),
                shape=np.append([0], psize),
                chunkshape=np.append([self.chunk_size], psize),
                filters=tb.Filters(complevel=self.complevel, complib=self.complib),
            )

            h5.create_earray(
                where=h5.root,
                name="areas",
                atom=tb.Int32Atom(),
                shape=np.append([0], psize),
                chunkshape=np.append([self.chunk_size], psize),
                filters=tb.Filters(complevel=self.complevel, complib=self.complib),
            )

        it = self.fnames_im
        if self.fnames_mask is not None:
            it = zip(self.fnames_im, self.fnames_mask)

        try:
            with tqdm(it, total=len(self.fnames_im)) as pbar:

                total_tiles = 0
                msg = msg if msg is not None else ""
                for fn in pbar:

                    # get the img and masks filenames
                    if self.fnames_mask is not None:
                        fn_im, fn_mask = fn
                    else:
                        fn_im = fn
                        fn_mask = None

                    pbar.set_description(f"Extracting {msg} patches to h5 db..")

                    # optionally patch and process images and masks
                    im, masks = self.get_array(
                        fn_im, fn_mask, tiling=tiling, pre_proc=pre_proc
                    )

                    n_tiles = im.shape[0] if tiling else 1

                    arg_list = []
                    for i in range(n_tiles):
                        kw = self._set_kwargs(im, fn_im, masks, fn_mask, tiling, i)
                        kw["h5"] = h5
                        arg_list.append(kw)

                    self._write_sequential(self.save2db, arg_list)
                    total_tiles += n_tiles
                    pbar.set_postfix_str(f"# of extracted tiles {total_tiles}")
        except Exception as e:
            h5.close()
            print(e)

        h5.root._v_attrs.n_items = h5.root.imgs.shape[0]
        h5.close()

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

        fn_im_new = fn
        fn_mask_new = None
        if mask_path is not None:
            fn_mask_new = fn

        if tiling:
            fn_im_new = f"{fn}_patch{ix + 1}"

            if mask_path is not None:
                fn_mask_new = f"{fn}_patch{ix + 1}"

        im_save_path = self.save_dir / fn_im_new

        mask_save_path = None
        if mask_path is not None:
            mask_save_path = self.save_dir / fn_mask_new

        kwarg = {"path_im": im_save_path, "path_mask": mask_save_path}

        if tiling:
            kwarg["image"] = im[ix : ix + 1]

            if mask_path is not None:
                for k, m in masks.items():
                    kwarg[k] = m[ix : ix + 1]
        else:
            kwarg["image"] = im[None, ...]
            if mask_path is not None:
                for k, m in masks.items():
                    kwarg[k] = m[None, ...]

        return kwarg

    def save2db(
        self,
        h5: BinaryIO,
        path_im: str,
        image: np.ndarray,
        path_mask: str = None,
        inst_map: np.ndarray = None,
        type_map: np.ndarray = None,
        sem_map: np.ndarray = None,
    ) -> None:
        """Write image and corresponding masks to h5-db."""
        h5.root.imgs.append(image)
        h5.root.fnames.append(np.array([path_im], dtype=str))

        if path_mask is not None:
            h5.root.insts.append(inst_map)

            if type_map is not None:
                h5.root.types.append(type_map)
            if sem_map is not None:
                h5.root.areas.append(sem_map)
