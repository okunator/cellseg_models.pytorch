# ONNX export

Export a trained CellPose or StarDist network to ONNX. Install `onnx` and
`onnxscript` to export, and an ONNX runtime to run the file.

The file contains prediction maps. Convert them to classes and instance masks
separately, using the same preprocessing as the checkpoint. Image height and
width are fixed; batch size can vary.

| Family | Input | Outputs, in order |
| --- | --- | --- |
| CellPose | `image`: floating `(B, 3, H, W)` | `flow_map`: `(B, 2, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |
| StarDist | `image`: floating `(B, 3, H, W)` | `binary_map`: foreground values `(B, 1, H, W)`; `ray_map`: `(B, n_rays, H, W)`; `type_map`: raw logits `(B, classes, H, W)` |

B is batch size, H height and W width. `out_size` can change the output dimensions.
The exporter uses the network's device and dtype and restores its training mode
afterward. It creates output folders and may overwrite the destination file.

```python
from pathlib import Path

import torch

from cellseg_models_pytorch.models.cellpose import CellPose

model = CellPose.from_pretrained(
    Path("checkpoints/cellpose.safetensors"), device=torch.device("cpu")
)
output_path = model.export_onnx(Path("exports/cellpose.onnx"))
```

## CellPose

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
