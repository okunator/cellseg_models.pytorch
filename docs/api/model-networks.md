# Model networks and factories

Create a PyTorch network for training. Start with a factory function such as
`cellpose_nuclei`; use a network class when you need custom outputs.
Set `enc_pretrain=False` to avoid downloading encoder weights.

```python
import torch

from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei

network = cellpose_nuclei(3, enc_name="resnet18", enc_pretrain=False)
network.eval()
with torch.inference_mode():
    output = network(torch.rand(1, 3, 64, 64))["nuc"]
```

This example uses an untrained network. `output.type_map` contains class scores
(logits) with shape `(1, 3, 64, 64)`. `output.aux_map` contains two flow maps with
shape `(1, 2, 64, 64)`. For image prediction and reconstruction, use a
[model class](models.md).

| Factory | Returned network | Maps used to separate objects | Separate foreground output |
| --- | --- | --- | --- |
| `cellpose_nuclei` | `CellPoseUnet` | 2 flows, vertical then horizontal | No |
| `omnipose_nuclei` | `CellPoseUnet` | 2 Omnipose flows, vertical then horizontal | No |
| `hovernet_nuclei` | `HoverNetUnet` | 2 position maps, vertical then horizontal targets | No |
| `cellvit_nuclei` | `CellVitSamUnet` | 2 position maps, vertical then horizontal targets | No |
| `stardist_nuclei` | `StarDistUnet` | `n_rays` radial distances | 1 channel |
| `instanseg_nuclei` | `InstanSegUnet` | 4 reconstruction channels | No |

Outputs use {py:class}`~cellseg_models_pytorch.decoders.multitask_decoder.SoftInstanceOutput`.
See [inference](inference.md) for the prediction fields.
Set `return_pred_only=False` if you also need encoder and decoder features.

For encoder freezing and a full training loop, see the
[training guide](../tutorials/training.ipynb).
Use standard [PyTorch module methods](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
to change training mode, move the network between devices and access its parameters.

## Configure a custom network

`decoders` names the decoder branches. `heads` names the outputs from each
branch and sets their channel counts.
Use `nuc_`, `cyto_` or `tissue_` prefixes and the reconstruction maps required by
your model family. Stage settings such as `out_channels` must match the network
depth. Changing the architecture requires compatible weights.

For CNN encoders such as `resnet18`, use image dimensions divisible by 32.
CellVit supports `samvit_*` encoders, which have their own image-size requirements.
For InstanSeg, use the factory or model class: these also create the pixel
classifier needed for reconstruction.

## CellPose family

```python
from cellseg_models_pytorch.models.cellpose.cellpose_unet import (
    CellPoseUnet,
    cellpose_nuclei,
    omnipose_nuclei,
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.cellpose.cellpose_unet.cellpose_nuclei
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.cellpose.cellpose_unet.omnipose_nuclei
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.cellpose.cellpose_unet.CellPoseUnet
   :members: forward
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.cellpose.cellpose_unet.CellPoseUnet.enc_name

   Encoder name used for access through ``network[network.enc_name]``.
```

## StarDist family

```python
from cellseg_models_pytorch.models.stardist.stardist_unet import (
    StarDistUnet,
    stardist_nuclei,
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.stardist.stardist_unet.stardist_nuclei
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.stardist.stardist_unet.StarDistUnet
   :members: forward
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.stardist.stardist_unet.StarDistUnet.enc_name

   Encoder name used for access through ``network[network.enc_name]``.
```

## HoverNet family

```python
from cellseg_models_pytorch.models.hovernet.hovernet_unet import (
    HoverNetUnet,
    hovernet_nuclei,
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.hovernet.hovernet_unet.hovernet_nuclei
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.hovernet.hovernet_unet.HoverNetUnet
   :members: forward
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.hovernet.hovernet_unet.HoverNetUnet.enc_name

   Encoder name used for access through ``network[network.enc_name]``.
```

## CellVit family

```python
from cellseg_models_pytorch.models.cellvit.cellvit_unet import (
    CellVitSamUnet,
    cellvit_nuclei,
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.cellvit.cellvit_unet.cellvit_nuclei
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.cellvit.cellvit_unet.CellVitSamUnet
   :members: forward
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.cellvit.cellvit_unet.CellVitSamUnet.enc_name

   Encoder name used for access through ``network[network.enc_name]``.
```

## InstanSeg family

```python
from cellseg_models_pytorch.models.instanseg import (
    InstanSegUnet,
    instanseg_nuclei,
)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.models.instanseg.instanseg_nuclei
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.instanseg.InstanSegUnet
   :members: forward
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.instanseg.InstanSegUnet.enc_name

   Encoder name used for access through ``network[network.enc_name]``.
```
