# Transforms

Normalize images before prediction, using the settings supplied with your
checkpoint. For training, use the target functions and classes below.

## Min-max normalization

```python
from cellseg_models_pytorch.transforms import minmax_normalize
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.minmax_normalize
```

## Other normalization functions

These functions accept NumPy images and return new arrays. Choose the scaling
used by your checkpoint. HW means height, width; HWC adds a channel axis.

| Function | Scaling |
| --- | --- |
| `normalize` | Subtract channel means and multiply by channel scales; no clipping |
| `percentile_normalize` | One pair of global percentiles, clipped to 0–1 |
| `percentile_normalize99` | Separate 1st/99th percentiles per channel; clipping optional |
| `float2ubyte` | Convert a floating image to uint8 for display/saving |

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.normalize
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.percentile_normalize
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.percentile_normalize99
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.float2ubyte
```

## Functions for training targets

Generate training targets from 2D instance masks: 0 is background and a positive
ID identifies each object. Apply image/mask augmentation first. These functions
keep the input mask unchanged.

| Function | Output | Meaning |
| --- | --- | --- |
| `gen_flow_maps` | Float64 `(2, H, W)` | Vertical then horizontal CellPose flows |
| `gen_omni_flow_maps` | Float64 `(2, H, W)` | Vertical then horizontal Omnipose flows |
| `gen_hv_maps` | Float64 `(2, H, W)` | Vertical then horizontal normalized positions |
| `gen_stardist_maps` | Float32 `(n_rays, H, W)` | Radial boundary distances in pixels |
| `gen_dist_maps` | Float64 `(H, W)` | Per-object distance, optionally normalized |
| `smooth_distance` | Float64 `(H, W)` | Smooth distance targets |
| `gen_contour_maps` | Float64 `(H, W)` | Boundary band |
| `gen_weight_maps` | Float64 `(H, W)` | Gap/border loss weights, zero inside objects |

H is height and W width. Empty masks return zeros. Contours can extend outside
objects; `gen_hv_maps` removes objects below `min_size`. For StarDist, pass the
model's ray count as `n_rays`.

```python
import numpy as np

from cellseg_models_pytorch.transforms import gen_hv_maps, gen_stardist_maps

instances = np.zeros((16, 16), dtype=np.int32)
instances[4:12, 4:12] = 1
position_target = gen_hv_maps(instances)
ray_target = gen_stardist_maps(instances, n_rays=8)
```

The target shapes are `(2, 16, 16)` and `(8, 16, 16)`. Convert them to tensors
with the dtype needed by your loss. Use the classes below to integrate target
generation with training datasets.

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_flow_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_omni_flow_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_hv_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_stardist_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_dist_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_contour_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.gen_weight_maps
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.transforms.functional.smooth_distance
```

## Image transforms for Compose

Install Albumentations to use these classes inside `albumentations.Compose`.
Pass an image with height, width, channels axes as `image=image`, then read the
result's `image` entry.

Import the classes from `cellseg_models_pytorch.transforms.albu_transforms`.

```python
import albumentations as A

from cellseg_models_pytorch.transforms.albu_transforms import (
    MinMaxNormalization,
)
from cellseg_models_pytorch.utils import FileHandler

normalization = A.Compose([MinMaxNormalization(copy=True)])
image = FileHandler.read_img("data/image.png")
normalized_image = normalization(image=image)["image"]
```

Normalization changes image values and leaves masks unchanged. Apply colour and intensity
changes to uint8 images before normalization.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.MinMaxNormalization
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.PercentileNormalization
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.Normalization
```

## Instance target classes

These classes generate training targets from instance masks. Augment images and
masks together first, then generate targets from the augmented masks.

| Class | Target name | Output axes | Meaning |
| --- | --- | --- | --- |
| `CellposeTransform` | `cellpose` | `(2, H, W)` | Vertical/horizontal flows |
| `HoverNetTransform` | `hovernet` | `(2, H, W)` | Vertical/horizontal position |
| `OmniposeTransform` | `omnipose` | `(2, H, W)` | Vertical/horizontal Omnipose flows |
| `StardistTransform` | `stardist` | `(n_rays, H, W)` | Radial boundary distances |
| `SmoothDistTransform` | `smoothdist` | `(H, W)` | Smooth interior distance |
| `DistTransform` | `dist` | `(H, W)` | Normalized interior distance |
| `ContourTransform` | `contour` | `(H, W)` | Object boundaries |
| `EdgeWeightTransform` | `edgeweight` | `(H, W)` | Weights for gaps between objects |
| `BinarizeTransform` | `binary` | `(H, W)` | Foreground/background |

H is height and W width. `name` supplies the target suffix; `out_channels` gives
the channel count. Outputs are float64, except StarDist rays (float32) and binary
masks (uint8).

`deduplicate=True` gives disconnected objects separate labels. CellposeTransform
currently ignores this option; supply distinct object labels yourself.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.CellposeTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.HoverNetTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.OmniposeTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.StardistTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.SmoothDistTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.DistTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.ContourTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.EdgeWeightTransform
   :members: __call__
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.BinarizeTransform
   :members: __call__
```

## Compose instance targets

`ApplyEach` generates several targets from the same mask. With `as_list=True`, it
returns a list of HWC arrays for the training datasets. Otherwise, it returns a
dictionary such as `{"inst-cellpose": ...}` with CHW or HW arrays.

```python
import numpy as np

from cellseg_models_pytorch.transforms.albu_transforms import (
    ApplyEach,
    BinarizeTransform,
)

instances = np.zeros((8, 8), dtype=np.int32)
instances[2:6, 2:6] = 1
targets = ApplyEach([BinarizeTransform()], as_list=True)(inst=instances)
```

`targets[0]` has shape `(8, 8, 1)`. Results follow transform order, then input
keyword order. See the [training guide](../tutorials/training.ipynb) for dataset use.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.ApplyEach
   :members: __call__
```

## Vary image appearance with StrongAugment

Use `AlbuStrongAugment` in Compose to vary image appearance while keeping masks
unchanged. Apply it to uint8 RGB images before normalization.
The parameter `probabilites` uses that spelling.

```python
import albumentations as A

from cellseg_models_pytorch.transforms.albu_transforms import (
    AlbuStrongAugment,
    MinMaxNormalization,
)
from cellseg_models_pytorch.utils import FileHandler

augmentation = A.Compose([
    AlbuStrongAugment(seed=42),
    MinMaxNormalization(copy=True),
])
image = FileHandler.read_img("data/image.png")
instances = FileHandler.read_mat("data/labels.mat")["inst"]
result = augmentation(image=image, masks=[instances])
```

`seed` controls operation choices and strengths. Some decisions also use NumPy's
random state; see the [training guide](../tutorials/training.ipynb) for repeatable
augmentation.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.transforms.albu_transforms.AlbuStrongAugment(augment_space=AUGMENT_SPACE, operations=(3, 4, 5), probabilites=(0.2, 0.3, 0.5), seed=None, p=1.0)
   :members: __call__
```
