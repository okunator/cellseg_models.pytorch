from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.torch_datasets import WSIDatasetInfer
from cellseg_models_pytorch.wsi import SlideReader
from cellseg_models_pytorch.wsi.inst_merger import InstMerger

__all__ = ["WsiSegmenter"]


class WsiSegmenter:
    def __init__(
        self,
        reader: SlideReader,
        model: BaseModelInst,
        level: int,
        coordinates: list[tuple[int, int, int, int]],
        batch_size: int = 8,
        normalization: Callable[..., dict[str, np.ndarray]] | None = None,
    ) -> None:
        """Class for segmenting WSIs.

        Args:
            reader: The `SlideReader` object for reading the WSIs.
            model: An instance segmentation model configured for inference.
            level: The level of the WSI to segment.
            coordinates:
                The bounding box coordinates from `reader.get_tile_coordinates()`.
            batch_size: The batch size for the DataLoader.
            normalization: Callable receiving ``image=tile`` and returning a
                dictionary containing the transformed HWC ``image``.
        """
        self.batch_size = batch_size
        self.coordinates = coordinates
        self.model = model

        self.dataset = WSIDatasetInfer(
            reader, coordinates, level=level, transforms=normalization
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self._has_processed = False

    def segment(self, save_dir: str | Path, maptype: str = "amap") -> None:
        """Segment the WSIs and save the instances as parquet files to `save_dir`.

        Args:
            save_dir: The directory to save the instances.
            maptype: Legacy output selector. Only ``amap`` (GeoParquet) is supported.

        Raises:
            ValueError: The requested map type is unsupported or tile paths collide.
            FileExistsError: A tile output already exists.

        Note:
            Uint8 RGB tiles are scaled to [0, 1]. Callable transforms return HWC
            values that are passed as float32 BCHW tensors on the model device.
            Failed runs retain any completed tile files and are not marked complete.
        """
        self._has_processed = False
        if maptype != "amap":
            raise ValueError("Only maptype='amap' (GeoParquet) is supported")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(self.dataloader, unit="batch") as loader:
            with torch.no_grad():
                for data in loader:
                    coords = [tuple(map(int, coord)) for coord in data["coords"]]
                    names = data["name"]
                    save_paths = [
                        save_dir / f"{name}_x{x}-y{y}_w{width}-h{height}.parquet"
                        for name, (x, y, width, height) in zip(names, coords)
                    ]
                    if len(set(save_paths)) != len(save_paths):
                        raise ValueError(
                            "Tile coordinates produce duplicate output paths"
                        )
                    for path in save_paths:
                        if path.exists():
                            raise FileExistsError(f"Tile output already exists: {path}")
                    im = (
                        data["image"]
                        .permute(0, 3, 1, 2)
                        .to(device=self.model.device, dtype=torch.float32)
                    )
                    if data["image"].dtype == torch.uint8:
                        im = im / 255

                    # predict
                    probs = self.model.predict(im)

                    # post-process
                    self.model.post_process(
                        probs,
                        save_paths_nuc=save_paths,
                        coords=coords,
                    )

        self._has_processed = True

    def merge_instances(
        self,
        src: str | Path,
        dst: str | Path,
        clear_in_dir: bool = False,
        simplify_level: float = 0.3,
        precision: int | None = None,
        *,
        filter_fragments: bool = False,
        filter_level: Literal["conservative", "middle", "strong"] = "middle",
        pixel_size: float | None = None,
    ) -> None:
        """Reconcile saved nucleus predictions using their source tiles.

        Reconcile duplicate and one-versus-many predictions in overlapping tile
        regions, keeping coherent source-tile observations and their attributes.
        Distinct nuclei may overlap without creating intersection objects.
        Prefer complete predictions; unmatched detections survive.
        Border flags recorded before smoothing distinguish
        partial contours from complete nuclei. Abutting fragments must actually
        touch to join, including in grids that also have overlapping tiles.
        Use overlapping tiles to predict complete contours across tile edges.

        Args:
            src:
                The directory containing the instances segmentations (.parquet-files).
            dst: The destination path for the output file. Allowed formats are
                '.parquet', '.geojson', and '.feather'.
            clear_in_dir: Whether to clear the source directory after merging.
                Defaults to False.
            simplify_level:
                The level of simplification to apply to the merged instances.
                Defaults to 0.3.
            precision:
                The precision level to apply to the merged instances. If None, no rounding
                is applied.
            filter_fragments: Remove small, thin polygons after stitching.
                Defaults to False. Retained contours and IDs stay unchanged.
            filter_level: Strength when filtering is enabled. Defaults to middle:
                area < 12 square micrometers AND thickness proxy
                (2 * area / perimeter) < 1.2 micrometers. Other levels are
                conservative (6, 0.8) and strong (20, 1.6). Small valid nuclei
                can also be removed; inspect the filtered result.
            pixel_size: Positive finite isotropic micrometers per level-zero
                coordinate pixel, required when filtering is enabled.
        """
        if not self._has_processed:
            raise ValueError("You must segment the instances first.")

        in_dir = Path(src)
        if not in_dir.is_dir():
            raise FileNotFoundError(in_dir)
        frames = []
        tile_indices = []
        source_tiles = {
            f"{self.dataset.reader.name}_x{x}-y{y}_w{width}-h{height}.parquet": index
            for index, (x, y, width, height) in enumerate(self.coordinates)
        }
        # The vector writer omits empty tiles; identify each saved tile by its name.
        for path in sorted(in_dir.glob("*.parquet")):
            if path.name not in source_tiles:
                raise ValueError(f"Prediction file has no source tile: {path.name}")
            frame = gpd.read_parquet(path)
            frames.append(frame)
            tile_indices.extend([source_tiles[path.name]] * len(frame))
        gdf = (
            gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
            if frames
            else gpd.GeoDataFrame(columns=["class_name", "geometry"])
        )
        merger = InstMerger(gdf, self.coordinates, tile_indices=tile_indices)
        merger.merge(
            dst,
            simplify_level=simplify_level,
            precision=precision,
            filter_fragments=filter_fragments,
            filter_level=filter_level,
            pixel_size=pixel_size,
        )

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()
