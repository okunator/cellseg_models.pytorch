# ONNX export

Export model prediction maps to ONNX.
From a source checkout, run `uv sync --locked --extra onnx` to install the
export packages. Install an ONNX runtime separately to run the file.

The file contains prediction maps. Convert them to classes and instance masks
separately, using the same preprocessing as the checkpoint. Image height and
width are fixed; batch size can vary.
The current export format requires a nucleus cell-type head. Binary-only models
can use PyTorch prediction and reconstruction, but are not supported by these exporters.

| Family | Input | Outputs, in order |
| --- | --- | --- |
| CellPose | `image`: floating `(B, 3, H, W)` | `flow_map`: `(B, 2, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| Omnipose | `image`: floating `(B, 3, H, W)` | `flow_map`: `(B, 2, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| StarDist | `image`: floating `(B, 3, H, W)` | `binary_map`: foreground values `(B, 1, H, W)`; `ray_map`: `(B, n_rays, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| HoverNet | `image`: floating `(B, 3, H, W)` | `hv_map`: horizontal/vertical distances `(B, 2, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| InstanSeg | `image`: floating `(B, 3, H, W)` | `instanseg_map`: reconstruction values `(B, 4, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| CellVit | `image`: floating `(B, 3, H, W)` | `hv_map`: horizontal/vertical distances `(B, 2, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |

B is batch size, H height and W width. `out_size` can change the output dimensions.
The exporter uses the network's device and dtype, even inside autocast, and restores
its training mode afterward. It creates output folders and may overwrite the destination file.

```python
from pathlib import Path

import torch

from cellseg_models_pytorch.models.cellpose import CellPose

model = CellPose.from_pretrained(
    Path("checkpoints/cellpose.safetensors"), device=torch.device("cpu")
)
output_path = model.export_onnx(Path("exports/cellpose.onnx"))
```

## CellPose and Omnipose

Omnipose uses `export_cellpose_onnx` too. Build its network with the same
architecture and class count as the training run, then load your weights:

```python
from cellseg_models_pytorch.models.cellpose import export_cellpose_onnx
from cellseg_models_pytorch.models.cellpose.cellpose_unet import omnipose_nuclei

network = omnipose_nuclei(3, enc_name="resnet18", enc_pretrain=False)
network.load_state_dict(
    torch.load("checkpoints/omnipose.pt", map_location="cpu", weights_only=True)
)
omnipose_path = export_cellpose_onnx(network, Path("exports/omnipose.onnx"))
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.cellpose.export_cellpose_onnx
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellpose.CellPose.export_onnx
```

## StarDist

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.stardist.export_stardist_onnx
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.stardist.StarDist.export_onnx
```

## HoverNet

Load your trained checkpoint, then export its prediction maps:

```python
from cellseg_models_pytorch.models.hovernet import HoverNet

hovernet = HoverNet.from_pretrained(
    Path("checkpoints/hovernet.pt"), device=torch.device("cpu")
)
hovernet_path = hovernet.export_onnx(Path("exports/hovernet.onnx"))
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.hovernet.export_hovernet_onnx

.. automethod:: cellseg_models_pytorch.models.hovernet.HoverNet.export_onnx
```

## InstanSeg

The four reconstruction channels contain two coordinate values, a scale value
and a seed value. They are raw network outputs. InstanSeg's pixel classifier stays
in Python; keep its trained weights with the model used for reconstruction.

```python
from cellseg_models_pytorch.models.instanseg import InstanSeg

instanseg = InstanSeg.from_pretrained(
    Path("checkpoints/instanseg.pt"), device=torch.device("cpu")
)
instanseg_path = instanseg.export_onnx(Path("exports/instanseg.onnx"))
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.instanseg.export_instanseg_onnx

.. automethod:: cellseg_models_pytorch.models.instanseg.InstanSeg.export_onnx
```

## CellVit

Use the SAM encoder and image size from training. Export is tested with
`samvit_base_patch16_224` at 224 × 224 pixels. Other SAM encoders and image sizes
need separate runtime checks.

```python
from cellseg_models_pytorch.models.cellvit import CellVit

cellvit = CellVit.from_pretrained(
    Path("checkpoints/cellvit.pt"), device=torch.device("cpu")
)
cellvit_path = cellvit.export_onnx(
    Path("exports/cellvit.onnx"), input_shape=(1, 3, 224, 224)
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.cellvit.export_cellvit_onnx

.. automethod:: cellseg_models_pytorch.models.cellvit.CellVit.export_onnx
```
