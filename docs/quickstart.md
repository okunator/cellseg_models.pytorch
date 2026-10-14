# Quick start

Turn an image into a mask that identifies each nucleus. You need an RGB image
and a local pretrained CellPose checkpoint. This example uses the HGSC checkpoint's
global min-max normalization; use the preprocessing specified for your weights.
Choose an image whose width and height are multiples of 32.

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with Image.open("data/image.png") as source:
    image = np.asarray(source.convert("RGB"))
image = minmax_normalize(image, copy=True)

model = CellPose.from_pretrained(Path("checkpoints/cellpose.safetensors"), device=device)
model.set_inference_mode(mixed_precision=False)
prediction = model.predict(image)
result = model.post_process(prediction, use_async_postproc=False)
instances, classes = result["nuc"][0]
```

`instances` has the same height and width as the image. Zero means background;
each positive number identifies one nucleus. `classes` gives each nucleus a
category number, whose meaning depends on the checkpoint.

For a runnable demonstration, output saving and larger images, continue with
[image inference](tutorials/inference.ipynb). To use your own labelled images,
start with [training](tutorials/training.ipynb).
