<div align="center">

![Logo](./images/logo.png)

**Python library for 2D cell/nuclei instance segmentation models written with [PyTorch](https://pytorch.org/).**

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/okunator/cellseg_models.pytorch/blob/master/LICENSE)
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.8.1+-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-red?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
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

## Features

- High level API to define cell/nuclei instance segmentation models.
- 6 cell/nuclei instance segmentation models and more to come.
- Open source datasets for training and benchmarking.
- Pre-trained backbones/encoders from the [timm](https://github.com/huggingface/pytorch-image-models) library.
- Pre-trained transformer backbones like [DinoV2](https://arxiv.org/abs/2304.07193) and [SAM](https://ai.facebook.com/research/publications/segment-anything/).
- All the architectures can be augmented to [panoptic segmentation](https://arxiv.org/abs/1801.00868).
- Flexibility to modify the components of the model architectures.
- Sliding window inference for large images.
- Multi-GPU inference.
- Popular training losses and benchmarking metrics.
- Benchmarking utilities both for model latency & segmentation performance.
- Regularization techniques to tackle batch effects/domain shifts such as [Strong Augment](https://arxiv.org/abs/2206.15274), [Spectral decoupling](https://arxiv.org/abs/2011.09468), [Label smoothing](https://arxiv.org/abs/1512.00567).
- Ability to add transformers to the decoder layers.
- Example notebooks to train models with [lightning](https://lightning.ai/docs/pytorch/latest/) or [accelerate](https://huggingface.co/docs/accelerate/index).

## Installation

**Basic installation**

```shell
pip install cellseg-models-pytorch
```

**To install extra dependencies (training utilities and datamodules for open-source datasets) use**

```shell
pip install cellseg-models-pytorch[all]
```

## Models

| Model                      | Paper                                                                          |
| -------------------------- | ------------------------------------------------------------------------------ |
| [[1](#Citation)] HoVer-Net | https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub |
| [[2](#Citation)] Cellpose  | https://www.nature.com/articles/s41592-020-01018-x                             |
| [[3](#Citation)] Omnipose  | https://www.biorxiv.org/content/10.1101/2021.11.03.467199v2                    |
| [[4](#Citation)] Stardist  | https://arxiv.org/abs/1806.03535                                               |
| [[5](#Citation)] CellVit-SAM  | https://arxiv.org/abs/2306.15350.03535                                               |
| [[6](#Citation)] CPP-Net  | https://arxiv.org/abs/2102.0686703535                                               |

## Datasets

| Dataset                       | Paper                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| [[7, 8](#References)] Pannuke | https://arxiv.org/abs/2003.10778 , https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2 |
| [[9](#References)] Lizard     | http://arxiv.org/abs/2108.11195                                                                  |

## Notebook examples

<details>
<summary style="margin-left: 25px;"> Training Hover-Net with Pannuke</summary>
<div style="margin-left: 25px;">

- [Training Hover-Net with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_hovernet.ipynb). Here we train the `Hover-Net` nuclei segmentation model with an `imagenet` pretrained `resnet50` backbone from the `timm` library. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained by utilizing [lightning](https://lightning.ai/docs/pytorch/latest/).

</div>
</details>

<details>
<summary style="margin-left: 25px;">Training Stardist with Pannuke</summary>
<div style="margin-left: 25px;">

- [Training Stardist with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_stardist.ipynb). Here we train the `Stardist` multi-class nuclei segmentation model with an `imagenet` pretrained `efficientnetv2_s` backbone from the `timm` library. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained by utilizing [lightning](https://lightning.ai/docs/pytorch/latest/).

</div>
</details>

<details>
<summary style="margin-left: 25px;">Training CellPose with Pannuke</summary>
<div style="margin-left: 25px;">

- [Training CellPose with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_cellpose.ipynb). Here we train the `CellPose` multi-class nuclei segmentation model with an `imagenet` pretrained `convnext_small` backbone from the `timm` library. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained (with checkpointing) by utilizing [accelerate](https://huggingface.co/docs/accelerate/index) by hugginface.

</div>
</details>

<details>
<summary style="margin-left: 25px;">Training OmniPose with Pannuke</summary>
<div style="margin-left: 25px;">

- [Training OmniPose with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_omnipose.ipynb). Here we train the OmniPose multi-class nuclei segmentation model with an `imagenet` pretrained `focalnet_small_lrf` backbone from the `timm` library. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained (with checkpointing) by utilizing [accelerate](https://huggingface.co/docs/accelerate/index) by hugginface.

</div>
</details>

<details>
<summary style="margin-left: 25px;">Training CPP-Net with Pannuke</summary>
<div style="margin-left: 25px;">

- [Training CPP-Net with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_cppnet.ipynb). Here we train the CPP-Net multi-class nuclei segmentation model with an `imagenet` pretrained `efficientnetv2_s` backbone from the `timm` library. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained by utilizing [lightning](https://lightning.ai/docs/pytorch/latest/).

</div>
</details>

<details>
<summary style="margin-left: 25px;">Finetuning CellPose with DINOv2 backbone</summary>
<div style="margin-left: 25px;">

- [Finetuning CellPose with DINOv2 backbone for Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_cellpose_dinov2.ipynb). Here we finetune the CellPose multi-class nuclei segmentation model with a `LVD-142M` pretrained `DINOv2` backbone. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained (with checkpointing) by utilizing [lightning](https://lightning.ai/docs/pytorch/latest/).

</div>
</details>

<details>
<summary style="margin-left: 25px;">Finetuning CellVit-SAM with Pannuke</summary>
<div style="margin-left: 25px;">

- [Finetuning CellVit-SAM with Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_nuclei_segmentation_cellvit.ipynb). Here we finetune the CellVit-SAM multi-class nuclei segmentation model with a `SA-1B` pretrained `SAM`-image-encoder backbone (checkout [`SAM`](https://github.com/facebookresearch/segment-anything)). The encoder is transformer based `VitDet`-model. The Pannuke dataset (fold 1 & fold 2) are used for training data and the fold 3 is used as validation data. The model is trained (with checkpointing) by utilizing [accelerate](https://huggingface.co/docs/accelerate/index) by hugginface.

</div>
</details>


<details>
<summary style="margin-left: 25px;">Benchmarking Cellpose Trained on Pannuke</summary>
<div style="margin-left: 25px;">

- [Benchmarking Cellpose Trained on Pannuke](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/pannuke_cellpose_benchmark.ipynb). Here we run benchmarking for `Cellpose` that was trained on Pannuke. Both the model performance and latency benchmarking are covered.

</div>
</details>

<details>
<summary style="margin-left: 25px;">Training CellPose with Lizard</summary>
<div style="margin-left: 25px;">

- [Training CellPose with Lizard](https://github.com/okunator/cellseg_models.pytorch/blob/main/examples/lizard_nuclei_segmentation_cellpose.ipynb). Here we train the `Cellpose` model with Lizard dataset that is composed of varying sized images. This example is old and might not be up to date.

</div>
</details>










## Code Examples

**Define Cellpose for cell segmentation.**

```python
import cellseg_models_pytorch as csmp
import torch

model = csmp.models.cellpose_base(type_classes=5)
x = torch.rand([1, 3, 256, 256])

# NOTE: the outputs still need post-processing.
y = model(x) # {"cellpose": [1, 2, 256, 256], "type": [1, 5, 256, 256]}
```

**Define Cellpose for cell and tissue area segmentation (Panoptic segmentation).**

```python
import cellseg_models_pytorch as csmp
import torch

model = csmp.models.cellpose_plus(type_classes=5, sem_classes=3)
x = torch.rand([1, 3, 256, 256])

# NOTE: the outputs still need post-processing.
y = model(x) # {"cellpose": [1, 2, 256, 256], "type": [1, 5, 256, 256], "sem": [1, 3, 256, 256]}
```

**Define panoptic Cellpose model with more flexibility.**

```python
import cellseg_models_pytorch as csmp

# the model will include two decoder branches.
decoders = ("cellpose", "sem")

# and in total three segmentation heads emerging from the decoders.
heads = {
    "cellpose": {"cellpose": 2, "type": 5},
    "sem": {"sem": 3}
}

model = csmp.CellPoseUnet(
    decoders=decoders,                   # cellpose and semantic decoders
    heads=heads,                         # three output heads
    depth=5,                             # encoder depth
    out_channels=(256, 128, 64, 32, 16), # num out channels at each decoder stage
    layer_depths=(4, 4, 4, 4, 4),        # num of conv blocks at each decoder layer
    style_channels=256,                  # num of style vector channels
    enc_name="resnet50",                 # timm encoder
    enc_pretrain=True,                   # imagenet pretrained encoder
    long_skip="unetpp",                  # unet++ long skips ("unet", "unetpp", "unet3p")
    merge_policy="sum",                  # concatenate long skips ("cat", "sum")
    short_skip="residual",               # residual short skips ("basic", "residual", "dense")
    normalization="bcn",                 # batch-channel-normalization.
    activation="gelu",                   # gelu activation.
    convolution="wsconv",                # weight standardized conv.
    attention="se",                      # squeeze-and-excitation attention.
    pre_activate=False,                  # normalize and activation after convolution.
)

x = torch.rand([1, 3, 256, 256])

# NOTE: the outputs still need post-processing.
y = model(x) # {"cellpose": [1, 2, 256, 256], "type": [1, 5, 256, 256], "sem": [1, 3, 256, 256]}
```

**Run HoVer-Net inference and post-processing with a sliding window approach.**

```python
import cellseg_models_pytorch as csmp

# define the model
model = csmp.models.hovernet_base(type_classes=5)

# define the final activations for each model output
out_activations = {"hovernet": "tanh", "type": "softmax", "inst": "softmax"}

# define whether to weight down the predictions at the image boundaries
# typically, models perform the poorest at the image boundaries and with
# overlapping patches this causes issues which can be overcome by down-
# weighting the prediction boundaries
out_boundary_weights = {"hovernet": True, "type": False, "inst": False}

# define the inferer
inferer = csmp.inference.SlidingWindowInferer(
    model=model,
    input_folder="/path/to/images/",
    checkpoint_path="/path/to/model/weights/",
    out_activations=out_activations,
    out_boundary_weights=out_boundary_weights,
    instance_postproc="hovernet",               # THE POST-PROCESSING METHOD
    normalization="percentile",                 # same normalization as in training
    patch_size=(256, 256),
    stride=128,
    padding=80,
    batch_size=8,
)

inferer.infer()

inferer.out_masks
# {"image1" :{"inst": [H, W], "type": [H, W]}, ..., "imageN" :{"inst": [H, W], "type": [H, W]}}
```

## Models API

Generally, the model building API enables the effortless creation of hard-parameter sharing multi-task encoder-decoder CNN architectures. The general architectural schema is illustrated in the below image.

<br><br>
![Architecture](./images/architecture_overview.png)

### Class API

The class API enables the most flexibility in defining different model architectures. It borrows a lot from [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) models API.

**Model classes**:

- `csmp.CellPoseUnet`
- `csmp.StarDistUnet`
- `csmp.HoverNet`
- `csmp.CellVitSAM`

**All of the models contain**:

- `model.encoder` - pretrained [timm](https://github.com/rwightman/pytorch-image-models) backbone for feature extraction.
- `model.{decoder_name}_decoder` - Models can have multiple decoders with unique names.
- `model.{head_name}_seg_head` - Model decoders can have multiple segmentation heads with unique names.
- `model.forward(x)` - forward pass.
- `model.forward_features(x)` - forward pass of the encoder and decoders. Returns enc and dec features

**Defining your own multi-task architecture**

For example, to define a multi-task architecture that has `resnet50` encoder, four decoders, and 5 output heads with `CellPoseUnet` architectural components, we could do this:

```python
import cellseg_models_pytorch as csmp
import torch

model = csmp.CellPoseUnet(
    decoders=("cellpose", "dist", "contour", "sem"),
    heads={
        "cellpose": {"type": 5, "cellpose": 2},
        "dist": {"dist": 1},
        "contour": {"contour": 1},
        "sem": {"sem": 4}
    },
)

x = torch.rand([1, 3, 256, 256])
model(x)
# {
#   "cellpose": [1, 2, 256, 256],
#   "type": [1, 5, 256, 256],
#   "dist": [1, 1, 256, 256],
#   "contour": [1, 1, 256, 256],
#   "sem": [1, 4, 256, 256]
# }
```

### Function API

With the function API, you can build models with low effort by calling the below listed functions. Under the hood, the function API simply calls the above classes with pre-defined decoder and head names. The training and post-processing tools of this library are built around these names, thus, it is recommended to use the function API, although, it is a bit more rigid than the class API. Basically, the function API only lacks the ability to define the output-tasks of the model, but allows for all the rest as the class API.

| Model functions                        | Output names                              | Task                             |
| -------------------------------------- | ----------------------------------------- | -------------------------------- |
| `csmp.models.cellpose_base`            | `"type"`, `"cellpose"`,                   | **instance segmentation**        |
| `csmp.models.cellpose_plus`            | `"type"`, `"cellpose"`, `"sem"`,          | **panoptic segmentation**        |
| `csmp.models.omnipose_base`            | `"type"`, `"omnipose"`                    | **instance segmentation**        |
| `csmp.models.omnipose_plus`            | `"type"`, `"omnipose"`, `"sem"`,          | **panoptic segmentation**        |
| `csmp.models.hovernet_base`            | `"type"`, `"inst"`, `"hovernet"`          | **instance segmentation**        |
| `csmp.models.hovernet_plus`            | `"type"`, `"inst"`, `"hovernet"`, `"sem"` | **panoptic segmentation**        |
| `csmp.models.hovernet_small`           | `"type"`,`"hovernet"`                     | **instance segmentation**        |
| `csmp.models.hovernet_small_plus`      | `"type"`, `"hovernet"`, `"sem"`           | **panoptic segmentation**        |
| `csmp.models.stardist_base`            | `"stardist"`, `"dist"`                    | **binary instance segmentation** |
| `csmp.models.stardist_base_multiclass` | `"stardist"`, `"dist"`, `"type"`          | **instance segmentation**        |
| `csmp.models.stardist_plus`            | `"stardist"`, `"dist"`, `"type"`, `"sem"` | **panoptic segmentation**        |
| `csmp.models.cppnet_base`              | `"stardist_refined"`, `"dist"`                    | **binary instance segmentation** |
| `csmp.models.cppnet_base_multiclass`   | `"stardist_refined"`, `"dist"`, `"type"`          | **instance segmentation**        |
| `csmp.models.cppnet_plus`              | `"stardist_refined"`, `"dist"`, `"type"`, `"sem"` | **panoptic segmentation**        |
| `csmp.models.cellvit_sam_base`         | `"type"`, `"inst"`, `"hovernet"`          | **instance segmentation**        |
| `csmp.models.cellvit_sam_plus`         | `"type"`, `"inst"`, `"hovernet"`, `"sem"` | **panoptic segmentation**        |
| `csmp.models.cellvit_sam_small`        | `"type"`,`"hovernet"`                     | **instance segmentation**        |
| `csmp.models.cellvit_sam_small_plus`   | `"type"`, `"hovernet"`, `"sem"`           | **panoptic segmentation**        |

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
