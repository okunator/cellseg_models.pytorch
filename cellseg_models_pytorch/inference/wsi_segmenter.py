from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.torch_datasets import WSIDatasetInfer
from cellseg_models_pytorch.utils import FileHandler
from cellseg_models_pytorch.utils.vectorize import gaussian_smooth, inst2gdf
from cellseg_models_pytorch.wsi import SlideReader
from cellseg_models_pytorch.wsi.inst_merger import InstMerger
from cellseg_models_pytorch.wsi.tiles import format_level

__all__ = ["WsiSegmenter"]


class WsiSegmenter:
    """Predict slide tiles and merge their saved nucleus polygons."""

    def __init__(
        self,
        reader: SlideReader,
        model: BaseModelInst,
        level: int,
        coordinates: list[tuple[int, int, int, int]],
        batch_size: int = 8,
        normalization: Callable[..., dict[str, np.ndarray | torch.Tensor]]
        | None = None,
        *,
        read_device: str | torch.device | None = None,
    ) -> None:
        """Set up tiled nucleus prediction for a slide.

        Args:
            reader: SlideReader used to read the tiles.
            model: Model with set_inference_mode already called. Use the normalization
                and class mapping supplied with its checkpoint.
            level: Slide level to read; 0 gives full resolution. Saved polygons
                always use level-zero slide coordinates.
            coordinates: List of level-zero (x, y, width, height) bounds. Pass
                TileCoordinates.coordinates when using reader-generated tiles.
                Read tiles in a batch must have matching image sizes.
            batch_size: Number of tiles predicted together.
            normalization: Callable receiving image=tile and returning a dictionary
                with an HWC image. Keep its size and pixel positions unchanged.
                For CUDA reads, accept and return a tensor.
            read_device: None reads CPU tiles. A CUDA device requires CUCIM and
                keeps tiles on GPU through normalization. Reconstruction and
                polygon saving still use CPU data.
        """
        self.batch_size = batch_size
        self.coordinates = coordinates
        self.model = model

        self.dataset = WSIDatasetInfer(
            reader,
            coordinates,
            level=level,
            transforms=normalization,
            read_device=read_device,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=read_device is None,
        )
        self._has_processed = False

    def segment(self, save_dir: str | Path, maptype: str = "amap") -> None:
        """Predict the selected tiles and save nucleus polygons.

        Args:
            save_dir: Tile-output folder; created if needed. Filenames record the
                slide name and source bounds. Choose a new folder for each run.
            maptype: Only "amap" (GeoParquet) is supported.

        Raises:
            ValueError: Map type is unsupported or source bounds produce duplicate names.
            FileExistsError: A tile output already exists.

        Note:
            Tiles are processed in batches as float32 tensors on the model device.
            Uint8 images are divided by 255; other values are kept as supplied.
            Saved polygons use the reader's level-zero pixel scale along each axis,
            including when the model returns smaller or larger prediction maps.
            Empty tiles produce no file. Failed runs keep completed tile files;
            merging is enabled only after every batch succeeds.
        """
        self._has_processed = False
        if maptype != "amap":
            raise ValueError("Only maptype='amap' (GeoParquet) is supported")
        down_y, down_x = 1.0, 1.0
        if self.dataset.level != 0:
            downsamples = self.dataset.reader.level_downsamples
            level = format_level(self.dataset.level, available=list(downsamples))
            down_y, down_x = downsamples[level]
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

                    # Reconstruct at prediction resolution before placing slide polygons.
                    masks = self.model.post_process(probs)
                    for (instances, classes), path, (x, y, _, _) in zip(
                        masks["nuc"], save_paths, coords
                    ):
                        polygons = inst2gdf(
                            instances,
                            classes,
                            xoff=x,
                            yoff=y,
                            smooth_func=gaussian_smooth,
                        )
                        xscale = down_x * im.shape[-1] / instances.shape[1]
                        yscale = down_y * im.shape[-2] / instances.shape[0]
                        if xscale != 1 or yscale != 1:
                            polygons.geometry = polygons.geometry.scale(
                                xfact=xscale, yfact=yscale, origin=(x, y)
                            )
                        FileHandler.gdf_to_file(polygons, path, silence_warnings=True)

        self._has_processed = True

    def merge_instances(
        self,
        src: str | Path,
        dst: str | Path,
        clear_in_dir: bool = False,
        simplify_level: float = 0.3,
        precision: int | None = None,
        *,
        filter_level: int | None = 0,
        pixel_size: float | None = None,
    ) -> None:
        """Merge this run's tile files into one slide-wide nucleus file.

        Args:
            src: Folder containing this run's nucleus .parquet files. Each filename
                must identify one of the original source tiles.
            dst: New .parquet, .geojson or .feather file. Parent folders are created.
            clear_in_dir: Delete every file in src and remove the folder after merging,
                including unrelated files. Leave False to retain tile outputs.
            simplify_level: Outline simplification tolerance in pixels; 0 disables it.
            precision: Decimal places for coordinate rounding, or None.
            filter_level: Small/thin-object filter after simplification and rounding.
                0 or None disables it. See filter_nuclei for levels and limits.
            pixel_size: Micrometers per level-zero pixel, required when filtering.
                Pixels must have equal physical height and width.

        Returns:
            None. Writes polygons and their selected source attributes, with new
            slide-wide uid values. See InstMerger.merge for merging behavior.

        Raises:
            ValueError: segment has not finished, a file has no matching source tile,
                or output/filter settings are invalid.
            FileNotFoundError: The source folder does not exist.
            FileExistsError: The destination already exists.

        Note:
            Merging loads the saved polygon tables into memory. Memory use grows
            with the number and complexity of polygons, even though reading and
            prediction use batches.
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
            filter_level=filter_level,
            pixel_size=pixel_size,
        )

        if clear_in_dir:
            for f in in_dir.glob("*"):
                f.unlink()
            in_dir.rmdir()
