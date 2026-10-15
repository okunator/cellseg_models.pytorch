# Quick start

Find individual nuclei in an RGB colour image using a pretrained CellPose model.
Download the [HGSC checkpoint](https://huggingface.co/csmp-hub/cellpose-histo-hgsc-nuc-v1)
and set the image and checkpoint paths below to your files. Choose an image whose
width and height are multiples of 32.

This checkpoint expects global min-max normalization, which rescales the image's
pixel values to 0–1. Use the preprocessing specified for your weights when choosing
a different checkpoint.

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
[image inference](tutorials/inference.ipynb). To train a model, start with
[training](tutorials/training.ipynb).
