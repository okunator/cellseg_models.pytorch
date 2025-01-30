from pathlib import Path
from typing import Callable, List, Tuple

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellseg_models_pytorch.inference import Inferer
from cellseg_models_pytorch.torch_datasets import WSIDatasetInfer
from cellseg_models_pytorch.wsi import SlideReader
from cellseg_models_pytorch.wsi.inst_merger import InstMerger

__all__ = ["WsiSegmenter"]


class WsiSegmenter:
    def __init__(
        self,
        inferer: Inferer,
        reader: SlideReader,
        level: int,
        coordinates: List[Tuple[int, int, int, int]],
        batch_size: int = 8,
        normalization: Callable = None,
    ) -> None:
        """Class for segmenting WSIs.

        Parameters:
            inferer (Inferer):
                The initialized Inferer object for segmenting the WSIs. Can be either
                `Inferer` or `SlidingWindowInferer`.
            reader (SlideReader):
                The `SlideReader` object for reading the WSIs.
            level (int):
                The level of the WSI to segment.
            coordinates (List[Tuple[int, int, int, int]]):
                The bounding box coordinates from `reader.get_tile_coordinates()`.
            batch_size (int):
                The batch size for the DataLoader.
            normalization (Callable):
                The normalization function for the DataLoader.
        """
        self.batch_size = batch_size
        self.coordinates = coordinates
        self.inferer = inferer

        self.dataset = WSIDatasetInfer(
            reader, coordinates, level=level, transform=normalization
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self._has_processed = False

    def segment(self, save_dir: str, maptype: str = "amap") -> None:
        """Segment the WSIs and save the instances as parquet files to `save_dir`.

        Parameters:
            save_dir (str):
                The directory to save the instances.
            maptype (str):
                The type of map to use for post-processing. Can be either 'amap','imap',
                'uimap', or 'map'.
        """
        save_dir = Path(save_dir)

        with tqdm(self.dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    im = (
                        data["image"]
                        .to(self.inferer.device)
                        .permute(0, 3, 1, 2)
                        .float()
                    )
                    coords = data["coords"]
                    names = data["name"]

                    # set args
                    save_paths = [
                        (save_dir / f"{n}_x{c[0]}-y{c[1]}_w{c[2]}-h{c[3]}").with_suffix(
                            ".parquet"
                        )
                        for n, c in zip(names, coords)
                    ]
                    coords = [tuple(map(int, coord)) for coord in coords]

                    # predict
                    probs = self.inferer.predict(im)

                    # post-process
                    self.inferer.post_process(
                        probs,
                        dst=save_paths,
                        coords=coords,
                        maptype=maptype,
                    )

        self._has_processed = True

    def merge_instances(self, src: str, dst: str, clear_in_dir: bool = False) -> None:
        """Merge the instances at the image boundaries.

        Parameters:
            src (str):
                The directory containing the instances segmentations (.parquet-files).
            dst (str):
                The destination path for the output file. Allowed formats are
                '.parquet', '.geojson', and '.feather'.
            clear_in_dir (bool, default=False):
                Whether to clear the source directory after merging.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        gdf = gpd.read_parquet(in_dir)
        merger = InstMerger(gdf, self.coordinates)
        merger.merge(dst)

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()
