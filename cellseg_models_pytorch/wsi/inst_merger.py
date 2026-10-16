from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import from_wkt, intersection, to_wkt, union_all
from shapely.geometry import box

__all__ = ["InstMerger"]


class InstMerger:
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        coordinates: list[tuple[int, int, int, int]],
        tile_indices: Sequence[int] | None = None,
    ) -> None:
        """Reconcile nucleus predictions in slide-pixel coordinates.

        Args:
            gdf: Nucleus polygons in level-zero pixel coordinates, with class_name.
            coordinates: Tile bounds in level-zero (x, y, width, height) pixels.
            tile_indices: Source-tile index for each input row. Required when tiles
                overlap; centroid position cannot identify the source tile.

        Raises:
            ValueError: Source indices are missing for overlapping tiles or invalid.
        """
        self.grid = gpd.GeoDataFrame(
            geometry=[box(x, y, x + w, y + h) for x, y, w, h in coordinates]
        )
        self.gdf = gdf.reset_index(drop=True)
        neighbors = self.grid.sindex.query(self.grid.geometry, predicate="intersects")
        self.neighbors = [(a, b) for a, b in zip(*neighbors) if a < b]
        self.overlapping = any(
            self.grid.geometry.iloc[a].intersection(self.grid.geometry.iloc[b]).area > 0
            for a, b in self.neighbors
        )
        # A slide location alone cannot identify which overlapping tile predicted it.
        if tile_indices is None:
            if self.overlapping:
                raise ValueError("Overlapping tiles require source tile_indices")
            self.owners = np.full(len(self.gdf), -1, dtype=int)
            matches = self.grid.sindex.query(
                self.gdf.geometry.representative_point(), predicate="within"
            )
            self.owners[matches[0]] = matches[1]
        else:
            indices = np.asarray(tile_indices)
            if (
                indices.shape != (len(self.gdf),)
                or (indices.size and indices.dtype.kind not in "iu")
                or np.any(indices < 0)
                or np.any(indices >= len(self.grid))
            ):
                raise ValueError(
                    "tile_indices must contain one valid tile index per row"
                )
            self.owners = indices.astype(int)

        if self.overlapping and not self.gdf.empty:
            # Prefer tile centres, where predictions have more surrounding image context.
            points = self.gdf.geometry.centroid
            rows, tiles = self.grid.sindex.query(points, predicate="within")
            centres = self.grid.geometry.centroid
            distances = (
                points.x.to_numpy()[rows] - centres.x.to_numpy()[tiles]
            ) ** 2 + (points.y.to_numpy()[rows] - centres.y.to_numpy()[tiles]) ** 2
            # Break equal-distance ties by tile index for reproducible ownership.
            order = np.lexsort((tiles, distances, rows))
            ordered_rows = rows[order]
            _, first = np.unique(ordered_rows, return_index=True)
            owner = np.full(len(self.gdf), -1, dtype=int)
            owner[ordered_rows[first]] = tiles[order[first]]
            # shortcut: ownership can drop valid detections; match objects before filtering.
            keep = self.owners == owner
            self.gdf = self.gdf.loc[keep].reset_index(drop=True)
            self.owners = self.owners[keep]

    def merge(
        self,
        dst: str | Path | None = None,
        simplify_level: float = 1,
        precision: int | None = None,
    ) -> gpd.GeoDataFrame | None:
        """Join neighboring tile fragments while retaining untouched nuclei.

        In overlap regions, retain predictions whose centroid belongs to their
        source tile's nearest-centre region. This can also discard valid detections
        when neighboring tiles predict different centroids or miss an object.
        Then match predictions covering at least half the smaller
        polygon. Greedy one-to-one matches cannot combine two predictions from the
        same source tile. Keep the original contour furthest from its tile edge,
        breaking ties by area; its attributes are retained too. No gap is filled.
        For abutting tiles, only actually touching fragments are unioned.
        Coordinates and CRS remain unchanged.

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
        candidates = []
        for left, right in self.neighbors:
            overlap = self.grid.geometry.iloc[left].intersection(
                self.grid.geometry.iloc[right]
            )
            if overlap.is_empty or (overlap.area == 0 and overlap.length == 0):
                continue
            indices = self.gdf.sindex.query(overlap, predicate="intersects")
            left_ids = indices[self.owners[indices] == left]
            right_ids = indices[self.owners[indices] == right]
            if not len(left_ids) or not len(right_ids):
                continue
            pairs = self.gdf.geometry.iloc[right_ids].sindex.query(
                self.gdf.geometry.iloc[left_ids], predicate="intersects"
            )
            a, b = left_ids[pairs[0]], right_ids[pairs[1]]
            geometries_a = self.gdf.geometry.iloc[a].to_numpy()
            geometries_b = self.gdf.geometry.iloc[b].to_numpy()
            shared = intersection(geometries_a, geometries_b)
            if overlap.area > 0:
                shared_areas = np.array([geometry.area for geometry in shared])
                # Compare with the smaller area to match partial and complete views.
                areas = np.minimum(
                    self.gdf.geometry.iloc[a].area.to_numpy(),
                    self.gdf.geometry.iloc[b].area.to_numpy(),
                )
                scores = np.divide(
                    shared_areas,
                    areas,
                    out=np.zeros(len(areas)),
                    where=areas > 0,
                )
                union_areas = (
                    self.gdf.geometry.iloc[a].area.to_numpy()
                    + self.gdf.geometry.iloc[b].area.to_numpy()
                    - shared_areas
                )
                ious = np.divide(
                    shared_areas,
                    union_areas,
                    out=np.zeros(len(areas)),
                    where=union_areas > 0,
                )
                candidates.extend(
                    (float(score), float(iou), int(i), int(j))
                    for score, iou, i, j in zip(scores, ious, a, b)
                    if score >= 0.5
                )
            elif not self.overlapping:
                graph.add_edges_from(
                    (i, j)
                    for i, j, geometry in zip(a, b, shared)
                    if geometry.length > 0
                )

        # Two nuclei predicted in the same tile must remain separate objects.
        groups = nx.utils.UnionFind()
        source_tiles = {}
        for _, _, left, right in sorted(
            candidates, key=lambda item: (-item[0], -item[1], item[2], item[3])
        ):
            a, b = groups[left], groups[right]
            if a == b:
                continue
            tiles_a = source_tiles.get(a, {self.owners[left]})
            tiles_b = source_tiles.get(b, {self.owners[right]})
            if tiles_a & tiles_b:
                continue
            groups.union(a, b)
            source_tiles[groups[left]] = tiles_a | tiles_b
            graph.add_edge(left, right)

        records = []
        consumed = np.zeros(len(self.gdf), dtype=bool)
        for component in nx.connected_components(graph):
            indices = sorted(component)
            fragments = self.gdf.iloc[indices]
            if self.overlapping:
                # Keep an observed contour with context instead of inventing an outline.
                context = [
                    self.gdf.geometry.iloc[i].distance(
                        self.grid.geometry.iloc[self.owners[i]].boundary
                    )
                    for i in indices
                ]
                best = max(
                    range(len(indices)),
                    key=lambda i: (context[i], fragments.geometry.iloc[i].area),
                )
                record = fragments.iloc[best].copy()
            else:
                # Abutting tiles can join only fragments that already share a boundary.
                record = fragments.iloc[
                    fragments.geometry.area.to_numpy().argmax()
                ].copy()
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
