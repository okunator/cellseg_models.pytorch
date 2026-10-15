from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import from_wkt, to_wkt, union_all
from shapely.geometry import box

__all__ = ["InstMerger"]


class InstMerger:
    def __init__(
        self, gdf: gpd.GeoDataFrame, coordinates: list[tuple[int, int, int, int]]
    ) -> None:
        """Merge polygons reconstructed from an abutting, nonoverlapping tile grid.

        Args:
            gdf: Nucleus polygons in level-zero pixel coordinates, with class_name.
            coordinates: Tile bounds in level-zero (x, y, width, height) pixels.
        """
        self.grid = gpd.GeoDataFrame(
            geometry=[box(x, y, x + w, y + h) for x, y, w, h in coordinates]
        )
        self.gdf = gdf.reset_index(drop=True)

    def merge(
        self,
        dst: str | Path | None = None,
        simplify_level: float = 1,
        precision: int | None = None,
    ) -> gpd.GeoDataFrame | None:
        """Join neighboring tile fragments while retaining untouched nuclei.

        Only polygons within two pixels of a shared tile edge can join. Connected
        fragments become one geometry; the largest fragment supplies attributes.
        Coordinates remain in the input pixel space, with the input CRS unchanged.

        Args:
            dst: New .parquet, .geojson or .feather file; None returns the result.
            simplify_level: Geometry simplification tolerance in pixels; zero disables it.
            precision: Optional number of decimal places for coordinate rounding.

        Returns:
            The merged GeoDataFrame when dst is None; otherwise writes the file
            and returns None. Each row has a unique slide-wide uid.

        Raises:
            ValueError: The destination extension is unsupported.
            FileExistsError: The destination already exists.
        """
        path = Path(dst) if dst is not None else None
        if path is not None:
            if path.suffix not in {".parquet", ".geojson", ".feather"}:
                raise ValueError(f"Unsupported output format: {path.suffix}")
            if path.exists():
                raise FileExistsError(path)

        graph = nx.Graph()
        if not self.gdf.empty and not self.grid.empty:
            owners = np.full(len(self.gdf), -1, dtype=int)
            matches = self.grid.sindex.query(
                self.gdf.geometry.representative_point(), predicate="within"
            )
            owners[matches[0]] = matches[1]
            neighbors = self.grid.sindex.query(self.grid.geometry, predicate="touches")
            for left, right in zip(*neighbors):
                if left >= right:
                    continue
                edge = self.grid.geometry.iloc[left].intersection(
                    self.grid.geometry.iloc[right]
                )
                if edge.length == 0:
                    continue
                candidates = self.gdf.sindex.query(
                    edge.buffer(2), predicate="intersects"
                )
                left_ids = candidates[owners[candidates] == left]
                right_ids = candidates[owners[candidates] == right]
                if not len(left_ids) or not len(right_ids):
                    continue
                # shortcut: two-pixel seam matching; use overlapping predictions for ambiguous fragments.
                pairs = self.gdf.geometry.iloc[right_ids].sindex.query(
                    self.gdf.geometry.iloc[left_ids], predicate="dwithin", distance=2
                )
                graph.add_edges_from(zip(left_ids[pairs[0]], right_ids[pairs[1]]))

        records = []
        consumed = np.zeros(len(self.gdf), dtype=bool)
        for component in nx.connected_components(graph):
            indices = sorted(component)
            fragments = self.gdf.iloc[indices]
            record = fragments.iloc[fragments.geometry.area.to_numpy().argmax()].copy()
            record["geometry"] = union_all(fragments.geometry.to_numpy())
            records.append(record)
            consumed[indices] = True
        merged = self.gdf.loc[~consumed].copy()
        if records:
            merged = gpd.GeoDataFrame(
                pd.concat(
                    [
                        merged,
                        gpd.GeoDataFrame(
                            records, geometry="geometry", crs=self.gdf.crs
                        ),
                    ],
                    ignore_index=True,
                ),
                geometry="geometry",
                crs=self.gdf.crs,
            )
        merged = merged.reset_index(drop=True)
        if simplify_level:
            merged.geometry = merged.geometry.simplify(simplify_level)
        if precision is not None:
            merged.geometry = from_wkt(
                to_wkt(merged.geometry.to_numpy(), rounding_precision=precision)
            )
        merged["uid"] = np.arange(len(merged))
        merged = merged.set_index("uid", drop=False)

        if path is None:
            return merged
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            merged.to_parquet(path)
        elif path.suffix == ".geojson":
            merged.to_file(path, driver="GeoJSON")
        else:
            merged.to_feather(path)
