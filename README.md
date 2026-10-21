<div align="center">

![Logo](./images/logo.png)

**Python library for 2D cell/nuclei instance segmentation models written with [PyTorch](https://pytorch.org/).**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/okunator/cellseg_models.pytorch/blob/main/LICENSE)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2+-red?&logo=pytorch)](https://pytorch.org/)
[![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-red?&logo=python&logoColor=white)](https://www.python.org/)
[![Model Checkpoints](https://img.shields.io/badge/HF-Model%20Hub-yellow)](https://huggingface.co/csmp-hub)
[![Tests](https://img.shields.io/github/actions/workflow/status/okunator/cellseg_models.pytorch/tests.yml?branch=main&label=Tests&logo=github)](https://github.com/okunator/cellseg_models.pytorch/actions/workflows/tests.yml)
<br>
[![Pypi](https://img.shields.io/pypi/v/cellseg-models-pytorch?color=blue&logo=pypi)](https://pypi.org/project/cellseg-models-pytorch/)
[![Downloads](https://api.pepy.tech/badge/cellseg-models-pytorch)](https://pepy.tech/projects/cellseg-models-pytorch)
[![Codecov](https://img.shields.io/codecov/c/github/okunator/cellseg_models.pytorch?logo=codecov)](https://codecov.io/gh/okunator/cellseg_models.pytorch)
[![DOI](https://zenodo.org/badge/450787123.svg)](https://zenodo.org/badge/latestdoi/450787123)
<br>

</div>

<div align="center">

</div>

## Introduction

**cellseg-models.pytorch** is a library built upon [PyTorch](https://pytorch.org/) that contains multi-task encoder-decoder architectures along with dedicated post-processing methods for segmenting cell/nuclei instances. See the [documentation](https://okunator.github.io/cellseg_models.pytorch/) for installation, user guides and the API reference. As the name might suggest, this library is heavily inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library for semantic segmentation.

For panoptic segmentation and downstream spatial analysis of whole-slide images,
see [Histolytics](https://hautaniemilab.github.io/histolytics/). Histolytics uses
cellseg-models.pytorch for its segmentation models.

## Features

- Six model families for cell and nuclei instance segmentation.
- Pretrained checkpoints available through [CSMP Hub](https://huggingface.co/csmp-hub).
- Configurable encoders, decoders and prediction heads.
- Cell-type classification and optional tissue segmentation alongside instance masks.
- Inference on individual images, sliding windows and whole-slide images.
- Datasets, transforms, losses and metrics for training and evaluation.
- ONNX export for all six model families.

## Installation

```shell
pip install cellseg-models-pytorch
```

Install optional features as needed:

```shell
pip install "cellseg-models-pytorch[training]"
pip install "cellseg-models-pytorch[onnx]"
pip install "cellseg-models-pytorch[openslide]"
```

Combine features with commas:

```shell
pip install "cellseg-models-pytorch[training,onnx]"
```

### Metrics only

Metrics with NumPy and SciPy, without PyTorch:

```shell
pip install cellseg-metrics
```

## Quick Start

**Initialize a pre-trained model and run inference**
```
pip install cellseg-models-pytorch
pip install albumentations==2.0.8
```

This checkpoint expects RGB images with pixel values scaled globally to 0–1.

```python
import numpy as np
from albumentations import Normalize, Resize

from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.utils import FileHandler

# Initialize a nuclei segmentation model trained on HGSC data.
# see models in https://huggingface.co/csmp-hub
model = CellPose.from_pretrained(weights="hgsc_v1_efficientnet_b5")

model.set_inference_mode()

# Resize to multiple of 32 of your own choosing
resize = Resize(1024, 1024)
normalize = Normalize(normalization="min_max")

im = FileHandler.read_img(IMG_PATH)
# Match the checkpoint's float32 preprocessing before normalization.
im = resize(image=im)["image"].astype(np.float32, copy=False)
im = normalize(image=im)["image"]

prob = model.predict(im)
out = model.post_process(prob)
# out = {"nuc": [(nuc instances (H, W), nuc types (H, W))], "cyto": None, "tissue": None}
```


## Models

Six model families are available. Find pretrained checkpoints on
[csmp-hub on Hugging Face](https://huggingface.co/csmp-hub).

| Model                      | Paper                                                                          |
| -------------------------- | ------------------------------------------------------------------------------ |
| [[1](REFERENCES.md#ref-1)] HoVer-Net | https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub |
| [[2](REFERENCES.md#ref-2)] Cellpose  | https://www.nature.com/articles/s41592-020-01018-x                             |
| [[3](REFERENCES.md#ref-3)] Omnipose  | https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2                    |
| [[4](REFERENCES.md#ref-4)] Stardist  | https://arxiv.org/abs/1806.03535                                               |
| [[5](REFERENCES.md#ref-5)] CellVit-SAM  | https://arxiv.org/abs/2306.15350                                               |
| [[6](REFERENCES.md#ref-6)] InstanSeg | https://arxiv.org/abs/2408.15954                                            |

## Datasets

Download prepared datasets from CSMP Hub. Each linked dataset card describes its
format, labels, splits and license.

| Dataset | Images | Splits | License |
| --- | ---: | --- | --- |
| [[7, 8](REFERENCES.md#ref-7)] [PanNuke](https://huggingface.co/datasets/csmp-hub/pannuke) | 7,901 | `fold1`, `fold2`, `fold3` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (noncommercial) |
| [[9](REFERENCES.md#ref-9)] [PanopTILs Refined](https://huggingface.co/datasets/csmp-hub/panoptils_refined) | 1,349 | Patient-separated `train`, `val`, `test` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [[10](REFERENCES.md#ref-10)] [PUMA](https://huggingface.co/datasets/csmp-hub/puma) | 206 | Public challenge `train` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [[11](REFERENCES.md#ref-11)] [CytoNuke](https://huggingface.co/datasets/csmp-hub/cytonuke) | 83 | New patient-separated `train`, `val`, `test` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (CPTAC images: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) |
| [[12](REFERENCES.md#ref-12)] [HeLaCytoNuc](https://huggingface.co/datasets/csmp-hub/helacytonuke) | 2,676 | Original `train`, `val` (automatic labels), `test` (manual labels) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [[13](REFERENCES.md#ref-13)] [AIR-LEISH](https://huggingface.co/datasets/csmp-hub/air-leish) | 180 | `train` only; no independent evaluation split | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [[14, 15](REFERENCES.md#ref-14)] [Lizard-Mitosis](https://huggingface.co/datasets/csmp-hub/lizard-mitosis) | 4,981 | Original `train`, `val`, `test`; train/val overlap | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (noncommercial) |
| [[16](REFERENCES.md#ref-16)] [Human Lung](https://huggingface.co/datasets/csmp-hub/human-lung) | 68 | Original `train` (51), `val` (7), `test` (10); patient independence unverified | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (noncommercial) |
| [[17, 18, 19](REFERENCES.md#ref-17)] [VitaminP ORION-CRC](https://huggingface.co/datasets/csmp-hub/vitaminp-orion-crc) | 3,337 | Original case-separated `train` (2,946), `val` (272), `test` (119) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Load one PanopTILs Refined sample:

```python
from io import BytesIO

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


def read_png(data: bytes) -> np.ndarray:
    """Decode RGB pixels or integer labels without reducing mask bit depth."""
    with Image.open(BytesIO(data)) as image:
        return np.array(image)


path = hf_hub_download(
    repo_id="csmp-hub/panoptils_refined",
    repo_type="dataset",
    filename="train.parquet",
)
row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
image = read_png(row["image"])
instances = read_png(row["inst"])
cell_types = read_png(row["type"])
tissue_types = read_png(row["sem"])
```

## References

See [REFERENCES.md](REFERENCES.md) for model papers and dataset sources.

## Citation

Please cite the version you used. [CITATION.cff](CITATION.cff) provides citation
metadata for the software as a whole. The example below cites version 0.1.30.

```bibtex
@misc{lehtonen_cellseg_models_2025,
  doi = {10.5281/zenodo.17608242},
  url = {https://doi.org/10.5281/zenodo.17608242},
  author = {Lehtonen, Oskari},
  title = {{cellseg_models.pytorch}},
  version = {0.1.30},
  publisher = {Zenodo},
  year = {2025}
}
```

## Licence

Original code in this project is licensed under [MIT](LICENSE).
Adapted third-party code retains its original terms. See
[third-party notices](NOTICE.md) for sources and bundled license texts,
including timm, Diffusers, Kornia, MetaFormer, HoVer-Net, Cellpose, StarDist,
scikit-image and HistoPrep.
Dataset and pretrained-weight licenses are documented separately in their cards.
