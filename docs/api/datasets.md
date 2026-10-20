# Datasets

Load training images and labels, or supply images for prediction.
Training datasets augment images and masks together, then generate training targets.
See the [training guide](../tutorials/training.ipynb) for a complete loop.
From a source checkout, `uv sync --locked --extra training` installs the required
augmentation and HDF5 packages.

## Choose the input layout

| Class | Input | Returned image | Other sample keys |
| --- | --- | --- | --- |
| `TrainDatasetFolder` | Parallel image and .mat label folders | `image`: CHW float32 tensor | Selected mask names; `cellpose_inst`, etc. |
| `TrainDatasetH5` | Configurable HDF5 root arrays | Configured image key: CHW float32 tensor | Selected masks; `inst_cellpose`, etc. |
| `FolderDatasetInfer` | Image folder | `image`: HWC RGB NumPy array, unless transformed | `name`: stem without extension |
| `HDF5DatasetInfer` | HDF5 `/imgs` and `/fnames` | `im`: CHW tensor retaining stored dtype | `file`: basename with extension |
| `WSIDatasetInfer` | Slide reader and coordinates | `image`: HWC NumPy array or tensor | `name`: slide name; `coords`: x, y, width, height |

H is height, W width and C channels. Sample keys differ between datasets, as shown
above. Training tensors start on CPU.

Instance labels use 0 for background and a positive ID for each object.
Class labels and normalization must match the model checkpoint.

## Training from folders

Image and label files are sorted separately and paired by position. Matching
filenames are not checked. Keep both folders aligned, with no extra files.
Each `.mat` file contains selected HW masks such as `inst` and `type`.

```python
import albumentations as A

from cellseg_models_pytorch.torch_datasets import TrainDatasetFolder
from cellseg_models_pytorch.transforms.albu_transforms import (
    ApplyEach,
    CellposeTransform,
    MinMaxNormalization,
)

dataset = TrainDatasetFolder(
    "data/train/images",
    "data/train/labels",
    mask_keys=("inst", "type"),
    transforms=A.Compose([MinMaxNormalization(copy=True)]),
    inst_transforms=ApplyEach([CellposeTransform()], as_list=True),
)
sample = dataset[0]
image = sample["image"]
cell_types = sample["type"].long()
flow_target = sample["cellpose_inst"].float()
```

`map_out_keys` selects and renames sample entries. Unlisted entries are dropped,
so include `image` if you need it. For example:
`{"image": "image", "type": "type_mask", "cellpose_inst": "cellpose"}`.
See [transforms](transforms.md) for augmentation and target options.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.torch_datasets.TrainDatasetFolder
   :members: __getitem__, __len__
```

## Training from HDF5

Install `tables` (PyTables) and Albumentations for this option. An existing HDF5
file must have root-level arrays with a shared first sample axis:

| Example node | Shape | Meaning |
| --- | --- | --- |
| `image` | `(N, H, W, 3)` | RGB images |
| `inst` | `(N, H, W)` | Instance labels |
| `type` | `(N, H, W)` | Cell-type class labels |

N is the number of samples. Node names are configurable. The dataset checks their
sample counts and reads one sample at a time.

```python
from cellseg_models_pytorch.torch_datasets import TrainDatasetH5

h5_dataset = TrainDatasetH5(
    "data/train.h5",
    img_key="image",
    inst_keys=("inst",),
    mask_keys=("inst", "type"),
    transforms=A.Compose([MinMaxNormalization(copy=True)]),
    inst_transforms=ApplyEach([CellposeTransform()], as_list=True),
)
h5_sample = h5_dataset[0]
flow_target = h5_sample["inst_cellpose"]
```

This example reuses the imports above. HDF5 target keys put the instance name first
(`inst_cellpose`); folder datasets use `cellpose_inst`. Labels are int64; images
and generated targets are float32. Set normalization through `transforms`.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.torch_datasets.TrainDatasetH5
   :members: __getitem__, __len__
```

## Folder inference

The optional transform receives the image directly: `transform(image)`.
Use a function such as `minmax_normalize`. For Albumentations Compose, call
`compose(image=image)["image"]` inside your transform.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.torch_datasets.FolderDatasetInfer
   :members: __getitem__, __len__
```

## HDF5 inference

```python
from cellseg_models_pytorch.torch_datasets.hdf5_dataset_infer import (
    HDF5DatasetInfer,
)
```

This reader expects `/imgs` and `/fnames`, unlike the configurable training reader.
It keeps stored pixel values and dtypes. Apply preprocessing before prediction.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.torch_datasets.hdf5_dataset_infer.HDF5DatasetInfer
   :members: __getitem__, __len__
```

## Slide-tile inference

See [slide reading](slide-reading.md) for levels and tile coordinates, or the
[WSI guide](../tutorials/wsi.ipynb) for the full workflow.
A normalization callback receives `image=tile` and returns an HWC `image`.
CUDA tiles require cuCIM and a callback that accepts tensors.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.torch_datasets.WSIDatasetInfer
   :members: __getitem__, __len__
```
