<div align="center">

![Logo](./images/logo.png)

**Python library for 2D cell/nuclei instance segmentation models written with [PyTorch](https://pytorch.org/).**

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg)](https://github.com/okunator/cellseg_models.pytorch/blob/master/LICENSE)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2+-red?&logo=pytorch)](https://pytorch.org/)
[![Python - Version](https://img.shields.io/badge/PYTHON-3.10+-red?&logo=python&logoColor=white)](https://www.python.org/)
[![Model Checkpoints](https://img.shields.io/badge/HF-Model%20Hub-yellow)](https://huggingface.co/csmp-hub)
<br>
[![Github Test](https://img.shields.io/github/actions/workflow/status/okunator/cellseg_models.pytorch/tests.yml?label=Tests&logo=github)](https://github.com/okunator/cellseg_models.pytorch/actions/workflows/tests.yml)
[![Pypi](https://img.shields.io/pypi/v/cellseg-models-pytorch?color=blue&logo=pypi)](https://pypi.org/project/cellseg-models-pytorch/)
[![Codecov](https://img.shields.io/codecov/c/github/okunator/cellseg_models.pytorch?logo=codecov)](https://codecov.io/gh/okunator/cellseg_models.pytorch)
[![DOI](https://zenodo.org/badge/450787123.svg)](https://zenodo.org/badge/latestdoi/450787123)
<br>

</div>

<div align="center">

</div>

## Introduction

**cellseg-models.pytorch** is a library built upon [PyTorch](https://pytorch.org/) that contains multi-task encoder-decoder architectures along with dedicated post-processing methods for segmenting cell/nuclei instances. As the name might suggest, this library is heavily inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library for semantic segmentation.

For panoptic segmentation and downstream spatial analysis of whole-slide images,
see [Histolytics](https://hautaniemilab.github.io/histolytics/). Histolytics uses
cellseg-models.pytorch for its segmentation models.

## Features

- High level API to define cell/nuclei instance segmentation models.
- 7 cell/nuclei instance segmentation model architectures
- Flexibility to modify the components of the model architectures.
- Sliding window inference for large images.

## Installation

```shell
pip install cellseg-models-pytorch
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

Find pretrained checkpoints on [csmp-hub on Hugging Face](https://huggingface.co/csmp-hub).

| Model                      | Paper                                                                          |
| -------------------------- | ------------------------------------------------------------------------------ |
| [[1](#references)] HoVer-Net | https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub |
| [[2](#references)] Cellpose  | https://www.nature.com/articles/s41592-020-01018-x                             |
| [[3](#references)] Omnipose  | https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2                    |
| [[4](#references)] Stardist  | https://arxiv.org/abs/1806.03535                                               |
| [[5](#references)] CellVit-SAM  | https://arxiv.org/abs/2306.15350                                               |
| [[6](#references)] CPP-Net  | https://arxiv.org/abs/2102.06867                                               |
| [[7](#references)] InstanSeg | https://arxiv.org/abs/2408.15954                                            |

## Datasets

| Dataset                       | Paper                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| [[8, 9](#references)] [PanNuke](https://huggingface.co/datasets/csmp-hub/pannuke) | https://arxiv.org/abs/2003.10778 , https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2 |

PanNuke is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for noncommercial use.

Download all three folds:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="csmp-hub/pannuke",
    repo_type="dataset",
    local_dir="data/pannuke-parquet",
    allow_patterns="*.parquet",
)
```

See the [dataset card](https://huggingface.co/datasets/csmp-hub/pannuke#read-a-sample)
for examples of reading the images and masks.

## References

- [1] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [2] Stringer, C.; Wang, T.; Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation Nature Methods, 2021, 18, 100-106
- [3] Cutler, K. J., Stringer, C., Wiggins, P. A., & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. bioRxiv. doi:10.1101/2021.11.03.467199
- [4] Uwe Schmidt, Martin Weigert, Coleman Broaddus, & Gene Myers (2018). Cell Detection with Star-Convex Polygons. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2018 - 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II (pp. 265–273).
- [5] Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J., Grünwald, B., Egger, J., & Kleesiek, J. (2023). CellViT: Vision Transformers for Precise Cell Segmentation and Classification (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2306.15350.
- [6] Chen, S., Ding, C., Liu, M., Cheng, J., & Tao, D. (2023). CPP-Net: Context-Aware Polygon Proposal Network for Nucleus Segmentation. In IEEE Transactions on Image Processing (Vol. 32, pp. 980–994). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tip.2023.3237013
- [7] Goldsborough, T., et al. (2024). InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation. arXiv. https://arxiv.org/abs/2408.15954.
- [8] Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019) PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In European Congress on Digital Pathology (pp. 11-19).
- [9] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A.,Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.

## Citation

```bibtex
@misc{https://doi.org/10.5281/zenodo.15357070,
  doi = {10.5281/ZENODO.15357070},
  url = {https://zenodo.org/doi/10.5281/zenodo.15357070},
  author = {Okunator,  },
  title = {okunator/cellseg_models.pytorch: v0.1.26},
  publisher = {Zenodo},
  year = {2025},
  copyright = {MIT License}
}
```

## Licence

This project is distributed under [MIT License](https://github.com/okunator/cellseg_models.pytorch/blob/main/LICENSE)

The project contains code from the original cell segmentation and 3rd-party libraries that have permissive licenses:

- [timm](https://github.com/rwightman/pytorch-image-models) (Apache-2)
- [HoVer-Net](https://github.com/vqdang/hover_net) (MIT)
- [Cellpose](https://github.com/MouseLand/cellpose) (BSD-3)
- [Stardist](https://github.com/stardist/stardist) (BSD-3)

If you find this library useful in your project, it is your responsibility to ensure you comply with the conditions of any dependent licenses. Please create an issue if you think something is missing regarding the licenses.
