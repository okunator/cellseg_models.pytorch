# Image prediction and instance masks

Use a [model class](models.md) for image prediction. Use `Predictor` to produce
prediction maps and `PostProcessor` to turn those maps into individual objects.
For whole slides, see [slide inference](wsi-inference.md) or the
[WSI guide](../tutorials/wsi.ipynb).

## Read prediction results

`Predictor` returns a dictionary with `nuc`, `cyto` and `tissue` entries.
Unused entries are None. Each `type_map` contains the predicted class at each pixel.

| Field | Direct network call | After Predictor |
| --- | --- | --- |
| `type_map` | Floating class logits, `(B, classes, H, W)` | Integer class IDs, `(B, H, W)` |
| `aux_map` | Flows or distances used to separate objects | Same maps, combined across image windows if requested |
| `binary_map` | Optional foreground map | Meaning depends on the model |
| `parents` | Names of the output branches | Unchanged |

B is batch size, H height and W width. Tensor maps stay on the network device.
`PostProcessor` converts them to CPU NumPy arrays.

Class IDs follow the checkpoint's labels. Class 0 can include untyped nuclei;
do not assume it always means background.

## Predictor

NumPy and PIL images use height, width, channels. Tensors use channels, height,
width, with an optional leading batch axis. Normalize images as required by the
checkpoint. Tensor inputs must match the network's device and floating dtype.

To keep raw class scores, call the network directly. `save_intermediate=True`
returns the same processed maps twice; it does not keep the raw scores.

```python
from cellseg_models_pytorch.inference.predictor import Predictor
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.inference.predictor.Predictor
   :members: predict_sliding_win
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.inference.predictor.Predictor.predict
```

## PostProcessor

```python
from cellseg_models_pytorch.inference.post_processor import PostProcessor
```

Choose the method matching your model: `cellpose`, `omnipose`, `hovernet` for
HoverNet/CellVit, or `stardist` for StarDist/CPPNet.

Batch methods require a `nuc` entry. Each result contains two int32 `(H, W)` masks:
`instances` identifies objects. `classes` gives the most common predicted class
within each object. Instance 0 is background; positive IDs identify objects
within that image.

```python
import torch

from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.transforms import minmax_normalize
from cellseg_models_pytorch.utils import FileHandler

model = CellPose.from_pretrained(
    "checkpoints/cellpose.safetensors", device=torch.device("cpu")
)
model.set_inference_mode(mixed_precision=False)
image = minmax_normalize(FileHandler.read_img("data/image.png"))
prediction = model.predict(image)
post_processor = PostProcessor("cellpose", {"use_gpu": False})
result = post_processor.postproc_serial(prediction)
instances, classes = result["nuc"][0]
```

Provide save paths to write polygons instead of returning masks. Paths may be
overwritten. Polygon coordinates are in pixels, with optional x/y offsets.
Smoothing changes saved polygon outlines only.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.inference.post_processor.PostProcessor
   :members: postproc_inst, postproc_tissuemap, postproc_serial, postproc_parallel, postproc_parallel_async
```

## Prediction map fields

Each container holds the maps for one output branch. Despite the "Soft" name,
`Predictor` replaces class scores with integer class labels.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.decoders.multitask_decoder.SoftInstanceOutput
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.decoders.multitask_decoder.SoftSemanticOutput
```
