from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

import numpy as np
import tables as tb
from tqdm import tqdm

from ._base_writer import BaseWriter

__all__ = ["SlidingWindowHDF5Writer"]


class SlidingWindowHDF5Writer(BaseWriter):
    def __init__(
        self,
        in_dir_im: str,
        in_dir_mask: str,
        save_dir: str,
        file_name: str,
        patch_size: Tuple[int, int],
        stride: int,
        transforms: Optional[List[str]] = None,
        chunk_size: int = 1,
        complevel: int = 5,
        complib: str = "blosc:lz4",
    ) -> None:
        """Write overlapping patches to a hdf5 database.

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
            >>> writer = SlidingWindowHDF5Writer(
                    "/path/to/my/imgs/,
                    "/path/to/my/masks/,
                    "/save/db/here/,
                    "my_h5_dataset.h5",
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

    def write(self) -> None:
        """Write patches to a hdf5 db."""
        h5 = tb.open_file(self.fname.as_posix(), mode="w")

        h5.create_earray(
            where=h5.root,
            name="imgs",
            atom=tb.UInt8Atom(),
            shape=np.append([0], self.patch_size + (3,)),
            chunkshape=np.append([self.chunk_size], self.patch_size + (3,)),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )

        h5.create_earray(
            where=h5.root,
            name="insts",
            atom=tb.Int32Atom(),
            shape=np.append([0], self.patch_size),
            chunkshape=np.append([self.chunk_size], self.patch_size),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )

        h5.create_earray(
            where=h5.root,
            name="types",
            atom=tb.Int32Atom(),
            shape=np.append([0], self.patch_size),
            chunkshape=np.append([self.chunk_size], self.patch_size),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )

        h5.create_earray(
            where=h5.root,
            name="areas",
            atom=tb.Int32Atom(),
            shape=np.append([0], self.patch_size),
            chunkshape=np.append([self.chunk_size], self.patch_size),
            filters=tb.Filters(complevel=self.complevel, complib=self.complib),
        )

        try:
            with tqdm(
                zip(self.fnames_im, self.fnames_mask), total=len(self.fnames_im)
            ) as pbar:
                total_tiles = 0
                for fni, fnm in pbar:
                    pbar.set_description("Extracting patches to h5 db..")
                    tiles = self._get_tiles(fni, fnm)
                    n_tiles = tiles["image"].shape[0]

                    arg_list = []
                    for i in range(n_tiles):
                        dd = dict(h5=h5)
                        for k, m in tiles.items():
                            dd[k] = m[i : i + 1]
                        arg_list.append(dd)

                    self._write_sequential(self.save2db, arg_list)
                    total_tiles += n_tiles
                    pbar.set_postfix_str(f"# of extracted tiles {total_tiles}")
        except Exception as e:
            h5.close()
            print(e)

        h5.root._v_attrs.n_items = h5.root.imgs.shape[0]
        h5.close()

    def save2db(
        self,
        h5: BinaryIO,
        image: np.ndarray,
        inst_map: np.ndarray,
        type_map: np.ndarray,
        sem_map: np.ndarray,
    ) -> None:
        """Write image and corresponding masks to h5-db."""
        h5.root.imgs.append(image)
        h5.root.insts.append(inst_map)
        h5.root.types.append(type_map)
        h5.root.areas.append(sem_map)
