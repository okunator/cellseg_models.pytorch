# File I/O

Use `FileHandler` to read images and labels or save prediction masks and polygons.

```python
from pathlib import Path

from cellseg_models_pytorch.utils import FileHandler
```

## Images and MATLAB labels

```python
image = FileHandler.read_img("data/image.png")
labels = FileHandler.read_mat("data/labels.mat")
instances = labels["inst"]
cell_types = labels["type"]
```

Images are RGB arrays in height, width, channels order, usually uint8 (0–255).
Select label arrays by their saved keys, such as `inst` and `type`.
For MATLAB v7.3 files, use an HDF5 reader.

```python
Path("outputs").mkdir(exist_ok=True)
FileHandler.to_mat(
    {"inst": instances, "type": cell_types},
    "outputs/prediction.mat",
)
```

Create the output folder first. Saving may overwrite an existing file.
`to_mat` saves only `inst`, `type`, `sem` and `cyto`; see its parameters below for
optional object measurements.

## Read existing HDF5 samples

Install `tables` (PyTables). Choose the array names and sample index to read:

```python
sample = FileHandler.read_h5("data/train.h5", 0, keys=("image", "inst", "type"))
```

For training, the [HDF5 dataset](datasets.md) reads these
samples and applies transforms.

## Save vector predictions

For whole-slide results, see [polygon merging](wsi-inference.md).

`gdf_to_file` saves a GeoDataFrame as `.parquet`, `.feather` or `.geojson`.
It keeps the input's coordinates and units. Empty data and unsupported suffixes
produce no file.

Create output folders first and choose new paths to preserve existing results.
Neither `gdf_to_file` nor `write_img` confirms that a file was written; check the
output when saving.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.utils.FileHandler
   :members: read_img, write_img, read_mat, to_mat, read_h5, gdf_to_file
```
