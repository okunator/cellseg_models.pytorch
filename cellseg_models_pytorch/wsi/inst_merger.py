from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.cg import alpha_shape_auto
from shapely.geometry import LineString, Polygon, box
from tqdm import tqdm

__all__ = ["InstMerger"]


class InstMerger:
    def __init__(
        self, gdf: gpd.GeoDataFrame, coordinates: List[Tuple[int, int, int, int]]
    ) -> None:
        """Merge instances at the boundaries of bbox coordinates.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The GeoDataFrame containing the non-merged instances.
            coordinates (List[Tuple[int, int, int, int]]):
                The bounding box coordinates from `reader.get_tile_coordinates()`.
        """
        # Convert xywh coordinates to bounding box polygons
        polygons = [box(x, y, x + w, y + h) for x, y, w, h in coordinates]
        self.grid = gpd.GeoDataFrame({"geometry": polygons})
        self.gdf = gdf

    def merge(
        self, dst: str = None, simplify_level: int = 1
    ) -> Union[gpd.GeoDataFrame, None]:
        """Merge the instances at the image boundaries.

        Parameters:
            dst (str):
                The destination directory to save the merged instances.
                If None, the merged GeoDataFrame is returned.
            simplify_level (int, default=1):
                The level of simplification to apply to the merged instances.

        Returns:
            Union[gpd.GeoDataFrame, None]:
                If `dst` is None, the merged GeoDataFrame is returned.

        Raises:
            ValueError:
                If the destination format is invalid.
        """
        if dst is not None:
            dst = Path(dst)
            suff = dst.suffix
            allowed_suff = [".parquet", ".geojson", ".feather"]
            if suff not in allowed_suff:
                raise ValueError(f"Invalid format. Got {suff}. Allowed: {allowed_suff}")

            parent = dst.parent
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        merge_obj_x, non_boundary_objs = self._merge_objs_axis(
            self.grid, self.gdf, axis="x", get_non_boundary_objs=True
        )
        merge_obj_y = self._merge_objs_axis(self.grid, self.gdf, axis="y")
        merged = pd.concat([merge_obj_x, merge_obj_y, non_boundary_objs]).reset_index(
            drop=True
        )
        merged.geometry = merged.geometry.simplify(simplify_level)

        if dst is not None:
            if suff == ".parquet":
                merged.to_parquet(dst)
            elif suff == ".geojson":
                merged.to_file(dst, driver="GeoJSON")
            elif suff == ".feather":
                merged.to_feather(dst)
        else:
            return merged

    def _merge_boundary_objs(
        self,
        boundary_objs: gpd.GeoDataFrame,
        midline_gdf: gpd.GeoDataFrame,
        objs: gpd.GeoDataFrame,
        apply_alpha_shape: bool = True,
        min_size: int = 25,
    ) -> gpd.GeoDataFrame:
        """Merge boundary objects that intersect the midline and touch each other."""
        boundary_objs = boundary_objs.loc[boundary_objs.is_valid]
        boundary_objs = boundary_objs.loc[boundary_objs.area > 10]

        # clip the midline at places where cells touch it
        touch = midline_gdf.clip(boundary_objs.union_all())
        touch = touch.explode()
        touch = touch.loc[touch.area > 5]

        # buffer the clipped midline and merge with the boundary objects
        touch_union = self._union_to_gdf(touch.union_all(), buffer_dist=1)
        merged = self._union_to_gdf(pd.concat([boundary_objs, touch_union]).union_all())

        if apply_alpha_shape:
            alpha_shapes = [
                alpha_shape_auto(
                    np.array(polygon.simplify(0.5).exterior.coords), step=2
                )
                for polygon in merged.geometry
            ]
            alpha_shape_gdf = gpd.GeoDataFrame(
                geometry=[poly for poly in alpha_shapes if poly.is_valid]
            )
            merged = self._union_to_gdf(
                pd.concat([merged, alpha_shape_gdf]).union_all()
            )

        if min_size:
            merged = merged.loc[merged.area > min_size]

        class_ids = self._get_classes(merged, objs)
        merged["class_name"] = class_ids

        return merged.reset_index(drop=True)

    def _get_objs(
        self,
        objects: gpd.GeoDataFrame,
        area: gpd.GeoDataFrame,
        predicate: str,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Get the objects that intersect with the midline."""
        inds = objects.geometry.sindex.query(
            area.geometry, predicate=predicate, **kwargs
        )
        objs: gpd.GeoDataFrame = objects.iloc[np.unique(inds)[1:]]

        return objs.drop_duplicates("geometry")

    def _merge_objs_axis(
        self,
        grid: gpd.GeoDataFrame,
        gdf: gpd.GeoDataFrame,
        axis: str = "x",
        midline_buffer: int = 2,
        predicate: str = "intersects",
        get_non_boundary_objs: bool = False,
    ) -> None:
        """Merge objects along the x or y axis."""
        if axis not in ["x", "y"]:
            raise ValueError("Axis must be either 'x' or 'y'")

        start_coord_key = "minx" if axis == "x" else "miny"
        ind = 0 if axis == "x" else 1

        grid[start_coord_key] = grid.geometry.apply(lambda geom: geom.bounds[ind])
        grid_sorted = grid.sort_values(by=start_coord_key)

        # Group by the 'minx'/'miny coordinate
        grouped = grid_sorted.groupby(start_coord_key)
        grouped_list = list(grouped)

        _, last_col = grouped_list[0]
        merged_gdfs = []
        midlines = []
        non_boundary_objs = []

        desc = "Merging objects (x-axis)" if axis == "x" else "Merging objects (y-axis)"
        for start, next_col in tqdm(grouped_list[1:], desc=desc):
            grid_union = pd.concat([last_col, next_col]).union_all()
            objs = self._get_objs(gdf, self._union_to_gdf(grid_union), predicate)

            minx, miny, maxx, maxy = next_col.total_bounds

            if axis == "x":
                midline = LineString([(start, miny), (start, maxy)]).buffer(
                    midline_buffer
                )
                midline_gdf = gpd.GeoDataFrame(geometry=[midline])
            else:
                midline = LineString([(minx, start), (maxx, start)]).buffer(
                    midline_buffer
                )
                midline_gdf = gpd.GeoDataFrame(geometry=[midline])

            # get the cells hitting the midline
            boundary_objs = self._get_objs(objs, midline_gdf, predicate)

            non_boundary_objs_left = None
            if get_non_boundary_objs:
                non_boundary_objs_left = self._get_objs(
                    objs, last_col.buffer(-midline_buffer), "contains"
                )

            # merge the boundary objects
            merged_boundary_objs = self._merge_boundary_objs(
                boundary_objs,
                midline_gdf,
                objs,
                apply_alpha_shape=True,
                min_size=25,
            )

            merged_gdfs.append(merged_boundary_objs)
            midlines.append(midline_gdf)
            non_boundary_objs.append(non_boundary_objs_left)
            last_col = next_col

        if get_non_boundary_objs:
            return pd.concat(merged_gdfs), pd.concat(non_boundary_objs)

        return pd.concat(merged_gdfs)

    def _union_to_gdf(self, union: Polygon, buffer_dist: int = 0) -> gpd.GeoDataFrame:
        """Convert a unionized GeoDataFrame back to a GeoDataFrame.

        Note: Fills in the holes in the polygons.
        """
        if isinstance(union, Polygon):
            union = gpd.GeoSeries([union.buffer(buffer_dist)])
        else:
            union = gpd.GeoSeries(
                [Polygon(poly.exterior).buffer(buffer_dist) for poly in union.geoms]
            )
        return gpd.GeoDataFrame(geometry=union)

    def _get_classes(
        self, merged: gpd.GeoDataFrame, non_merged: gpd.GeoDataFrame
    ) -> None:
        """Get the class ids for the merged objs from the non-merged objs w/ majority voting."""
        class_names = []
        for ix, row in merged.iterrows():
            area = gpd.GeoDataFrame(geometry=[row.geometry])
            objs = self._get_objs(non_merged, area, predicate="intersects")

            if objs.empty:
                continue

            class_names.append(objs.loc[objs.area.idxmax()]["class_name"])

        return class_names
