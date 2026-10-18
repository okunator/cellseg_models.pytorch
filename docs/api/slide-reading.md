# Slide reading

Open a slide, find tissue and choose regions to predict. For a complete workflow,
see the [whole-slide guide](../tutorials/wsi.ipynb).

## Open a slide

```python
from cellseg_models_pytorch.wsi import SlideReader

reader = SlideReader("data/slide.svs", backend="OPENSLIDE")
level = reader.level_from_max_dimension(1024)
thumbnail = reader.read_level(level)
```

Choose a reader installed in your environment:

| Backend | Required software |
| --- | --- |
| `OPENSLIDE` | OpenSlide Python bindings and native library |
| `CUCIM` | cuCIM; CUDA tile reads also need CuPy and a compatible GPU |
| `BIOIO` | BioIO and a reader plugin for your format |

Use OpenSlide or cuCIM for this workflow. BioIO's whole-level reads currently
need correction.

## Choose tiles

```python
threshold, tissue_mask = reader.get_tissue_mask(level=level)
tiles = reader.get_tile_coordinates(
    width=256,
    tissue_mask=tissue_mask,
    overlap=0.25,
    out_of_bounds=False,
)
tile = reader.read_region(tiles[0], level=0)
preview = reader.get_annotated_thumbnail(thumbnail, tiles)
properties = tiles.get_properties(
    level=0, level_downsample=reader.level_downsamples[0]
)
```

This assumes at least one tissue tile is found. Pass `tissue_mask=None` to keep
background tiles. Use a mask scaled equally along both axes.

Dimensions use **(height, width)**. Region bounds use **(x, y, width, height)**
in full-resolution pixels, with (0, 0) at the top left. Reading another level
reduces the returned image size; keep the region bounds in level-zero pixels.

`pixel_size` gives physical height and width per level-zero pixel, in micrometers.
Reading `pixel_size` raises an error if this information is missing.
You can still read tiles without it.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.wsi.SlideReader
   :members: path, name, suffix, backend_name, data_bounds, dimensions, pixel_size, level_count, level_dimensions, level_downsamples, read_level, read_region, level_from_max_dimension, level_from_dimensions, get_tissue_mask, get_tile_coordinates, get_spot_coordinates, get_annotated_thumbnail
```

## Coordinate collections

```python
from cellseg_models_pytorch.wsi.data import SpotCoordinates, TileCoordinates
```

Index or iterate over either collection to get region bounds. Use
`tiles.coordinates` when passing a list to [WsiSegmenter](wsi-inference.md).

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.wsi.data.TileCoordinates
   :members: __len__, __iter__, __getitem__, get_properties

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.coordinates

   Tile bounds in reading order, as level-zero (x, y, width, height) tuples.

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.width

   Tile width in level-zero pixels.

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.height

   Tile height in level-zero pixels.

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.overlap

   Fraction shared by neighboring tiles.

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.max_background

   Largest allowed background fraction, or None when filtering was disabled.

.. py:attribute:: cellseg_models_pytorch.wsi.data.TileCoordinates.tissue_mask

   Mask used to choose tiles, or None.
```

For tissue microarrays, detect spots from a cleaned tissue mask:

```python
spots = reader.get_spot_coordinates(tissue_mask)
named_spots = list(zip(spots.spot_names, spots))
```

Names and bounds have the same order. Iterating over `spots` returns bounds only.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.wsi.data.SpotCoordinates
   :members: __len__, __iter__, __getitem__

.. py:attribute:: cellseg_models_pytorch.wsi.data.SpotCoordinates.coordinates

   Spot bounds as a list of level-zero (x, y, width, height) tuples.

.. py:attribute:: cellseg_models_pytorch.wsi.data.SpotCoordinates.spot_names

   Names in the same order as coordinates.

.. py:attribute:: cellseg_models_pytorch.wsi.data.SpotCoordinates.tissue_mask

   Cleaned tissue mask at its detection resolution.
```
