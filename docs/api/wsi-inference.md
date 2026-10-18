# Slide inference and merging

Predict slide tiles, then combine their nucleus polygons into one result.
See the [whole-slide guide](../tutorials/wsi.ipynb) for the full workflow and plots.

## Predict and merge tiles

```python
import albumentations as A
import geopandas as gpd
import torch

from cellseg_models_pytorch.inference.post_processor import (
    PostProcessor,
)
from cellseg_models_pytorch.inference.wsi_segmenter import (
    WsiSegmenter,
)
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.transforms.albu_transforms import (
    MinMaxNormalization,
)
from cellseg_models_pytorch.wsi import SlideReader

reader = SlideReader("data/slide.svs", backend="OPENSLIDE")
tiles = reader.get_tile_coordinates(
    width=256,
    tissue_mask=None,
    overlap=0.25,
    out_of_bounds=False,
)
model = CellPose.from_pretrained(
    "checkpoints/cellpose.safetensors", device=torch.device("cpu")
)
model.set_inference_mode(mixed_precision=False)
model.post_processor = PostProcessor("cellpose", {"use_gpu": False})
segmenter = WsiSegmenter(
    reader,
    model,
    level=0,
    coordinates=tiles.coordinates,
    batch_size=1,
    normalization=A.Compose([MinMaxNormalization(copy=True)]),
)
segmenter.segment("outputs/slide-tiles")
segmenter.merge_instances(
    "outputs/slide-tiles", "outputs/slide-nuclei.parquet", simplify_level=0
)
nuclei = gpd.read_parquet("outputs/slide-nuclei.parquet")
```

Use your checkpoint's normalization. Choose new output paths for each run.
Empty tiles produce no file; saved filenames identify the source tile for merging.
Use `level=0`: higher-level predictions are not yet scaled back to slide coordinates.

`nuclei` contains one row per object, with its polygon, class and slide-wide `uid`.
Coordinates remain in pixels. See [metrics](metrics.md) to evaluate masks and the
[WSI guide](../tutorials/wsi.ipynb) to plot saved polygons.

Prediction reads images in batches. Merging loads the saved polygon tables into
memory, so memory use grows with the number and complexity of objects.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.inference.wsi_segmenter.WsiSegmenter
   :members: segment, merge_instances
```

## Merge polygons directly

Use `InstMerger` when you already have a GeoDataFrame of predictions. Keep each
polygon's source-tile index, especially for overlapping tiles.

This small example describes the same nucleus in two tiles, plus a separate cell:

```python
import geopandas as gpd
from shapely.geometry import box

from cellseg_models_pytorch.wsi.inst_merger import InstMerger, filter_nuclei

bounds = [(0, 0, 64, 64), (32, 0, 64, 64)]
predictions = gpd.GeoDataFrame(
    {"class_name": ["tumor", "tumor", "immune"]},
    geometry=[box(40, 20, 48, 28), box(40, 20, 48, 28), box(80, 20, 88, 28)],
)
merger = InstMerger(predictions, bounds, tile_indices=[0, 1, 1])
merged = merger.merge(simplify_level=0)
```

`merged` has two objects. Duplicate predictions keep a whole contour from one
tile; unmatched cells remain. Touching fragments can join across a shared tile
edge. Other attributes come from the selected source, or the largest joined fragment.

Pass `dst` to save instead of returning a table. Supported formats are `.parquet`,
`.geojson` and `.feather`; existing files are not overwritten.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.wsi.inst_merger.InstMerger
   :members: merge
```

## Optional filtering

Filtering is off by default. It removes objects only when they are both small
and thin; valid small nuclei can also be removed. Supply a measured pixel size,
with equal physical height and width. The limits are listed below.

For the example polygons, assuming 0.25 micrometers per pixel:

```python
filtered = filter_nuclei(merged, pixel_size=0.25, filter_level=1)
```

Filtering keeps surviving outlines and IDs. To filter while merging, pass
`filter_level` and `pixel_size` to either merge method.

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.wsi.inst_merger.filter_nuclei
```
