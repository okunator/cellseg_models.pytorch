<div align="center">

![Logo](./images/logo.png)

**Python library for 2D cell/nuclei instance segmentation models written with [PyTorch](https://pytorch.org/).**

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/okunator/cellseg_models.pytorch/blob/master/LICENSE)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python - Version](https://img.shields.io/badge/PYTHON-3.10+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
<br>
[![Github Test](https://img.shields.io/github/actions/workflow/status/okunator/cellseg_models.pytorch/tests.yml?label=Tests&logo=github&&style=for-the-badge)](https://github.com/okunator/cellseg_models.pytorch/actions/workflows/tests.yml)
[![Pypi](https://img.shields.io/pypi/v/cellseg-models-pytorch?color=blue&logo=pypi&style=for-the-badge)](https://pypi.org/project/cellseg-models-pytorch/)
[![Codecov](https://img.shields.io/codecov/c/github/okunator/cellseg_models.pytorch?logo=codecov&style=for-the-badge&token=oGSj7FZ1lm)](https://codecov.io/gh/okunator/cellseg_models.pytorch)
<br>
[![DOI](https://zenodo.org/badge/450787123.svg)](https://zenodo.org/badge/latestdoi/450787123)

</div>

<div align="center">

</div>

## Introduction

**cellseg-models.pytorch** is a library built upon [PyTorch](https://pytorch.org/) that contains multi-task encoder-decoder architectures along with dedicated post-processing methods for segmenting cell/nuclei instances. As the name might suggest, this library is heavily inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library for semantic segmentation. 

## What's new? 📢
- Simplified model usage
- First pre-trained models in [csmp-hub](https://huggingface.co/csmp-hub). More to come soon!

## Features 🌟

- High level API to define cell/nuclei instance segmentation models.
- 6 cell/nuclei instance segmentation model architectures
- Open source datasets for training and benchmarking.
- Flexibility to modify the components of the model architectures.
- Sliding window inference for large images.
- Popular training losses and benchmarking metrics.
- Benchmarking utilities
- Regularization techniques to tackle batch effects/domain shifts such as [Strong Augment](https://arxiv.org/abs/2206.15274), [Spectral decoupling](https://arxiv.org/abs/2011.09468), [Label smoothing](https://arxiv.org/abs/1512.00567).


## Installation 🛠️

```shell
pip install cellseg-models-pytorch
```

## Models 🤖

| Model                      | Paper                                                                          |
| -------------------------- | ------------------------------------------------------------------------------ |
| [[1](#Citation)] HoVer-Net | https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub |
| [[2](#Citation)] Cellpose  | https://www.nature.com/articles/s41592-020-01018-x                             |
| [[3](#Citation)] Omnipose  | https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2                    |
| [[4](#Citation)] Stardist  | https://arxiv.org/abs/1806.03535                                               |
| [[5](#Citation)] CellVit-SAM  | https://arxiv.org/abs/2306.15350                                               |
| [[6](#Citation)] CPP-Net  | https://arxiv.org/abs/2102.06867                                               |

## Datasets

| Dataset                       | Paper                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| [[7, 8](#References)] Pannuke | https://arxiv.org/abs/2003.10778 , https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2 |


## Quick Start 💻

**Initialize a pre-trained model and run inference**
```
pip install cellseg-models-pytorch
pip install albumentations
```

```python
from albumentations import Resize, Compose
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.utils import FileHandler
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

# intialize nuclei segmentation model trained on HGSC data
# see models in https://huggingface.co/csmp-hub
model = CellPose.from_pretrained(weights="hgsc_v1_efficientnet_b5")

model.set_inference_mode()

# Resize to multiple of 32 of your own choosing
transform = Compose([Resize(1024, 1024), MinMaxNormalization()])

im = FileHandler.read_img(IMG_PATH)
im = transform(image=im)["image"]

prob = model.predict(im)
out = model.post_process(prob)
# out = {"nuc": [(nuc instances (H, W), nuc types (H, W))], "cyto": None, "tissue": None}
```


## References

- [1] S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019.
- [2] Stringer, C.; Wang, T.; Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation Nature Methods, 2021, 18, 100-106
- [3] Cutler, K. J., Stringer, C., Wiggins, P. A., & Mougous, J. D. (2022). Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation. bioRxiv. doi:10.1101/2021.11.03.467199
- [4] Uwe Schmidt, Martin Weigert, Coleman Broaddus, & Gene Myers (2018). Cell Detection with Star-Convex Polygons. In Medical Image Computing and Computer Assisted Intervention - MICCAI 2018 - 21st International Conference, Granada, Spain, September 16-20, 2018, Proceedings, Part II (pp. 265–273).
- [5] Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J., Grünwald, B., Egger, J., & Kleesiek, J. (2023). CellViT: Vision Transformers for Precise Cell Segmentation and Classification (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2306.15350.
- [6] Chen, S., Ding, C., Liu, M., Cheng, J., & Tao, D. (2023). CPP-Net: Context-Aware Polygon Proposal Network for Nucleus Segmentation. In IEEE Transactions on Image Processing (Vol. 32, pp. 980–994). Institute of Electrical and Electronics Engineers (IEEE). https://doi.org/10.1109/tip.2023.3237013
- [7] Gamper, J., Koohbanani, N., Benet, K., Khuram, A., & Rajpoot, N. (2019) PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In European Congress on Digital Pathology (pp. 11-19).
- [8] Gamper, J., Koohbanani, N., Graham, S., Jahanifar, M., Khurram, S., Azam, A.,Hewitt, K., & Rajpoot, N. (2020). PanNuke Dataset Extension, Insights and Baselines. arXiv preprint arXiv:2003.10778.
- [9] Graham, S., Jahanifar, M., Azam, A., Nimir, M., Tsang, Y.W., Dodd, K., Hero, E., Sahota, H., Tank, A., Benes, K., & others (2021). Lizard: A Large-Scale Dataset for Colonic Nuclear Instance Segmentation and Classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 684-693).

## Citation

```bibtex
@misc{csmp2022,
    title={{cellseg_models.pytorch}: Cell/Nuclei Segmentation Models and Benchmark.},
    author={Oskari Lehtonen},
    howpublished = {\url{https://github.com/okunator/cellseg_models.pytorch}},
    doi = {10.5281/zenodo.7064617}
    year={2022}
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
