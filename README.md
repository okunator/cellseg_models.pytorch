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

Download prepared datasets from CSMP Hub. Each Parquet row contains an image
(`image`), nucleus instances (`inst`) and nucleus classes (`type`). Masks use PNG
bytes; images use RGB PNG bytes except Human Lung, which keeps four float32
channels as TIFF bytes. PanopTILs Refined, PUMA and Human Lung also include
semantic labels (`sem`) for tissue and other regions.
CytoNuke, HeLaCytoNuc, AIR-LEISH and Human Lung include separate whole-cell instances
(`inst_cell`) and inferred nucleus-to-cell links (`pairs`). AIR-LEISH also
includes parasite instances (`inst_parasite`) and full masks for overlapping objects.

| Dataset | Images | Splits | License |
| --- | ---: | --- | --- |
| [[8, 9](#references)] [PanNuke](https://huggingface.co/datasets/csmp-hub/pannuke) | 7,901 | `fold1`, `fold2`, `fold3` | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (noncommercial) |
| [[10](#references)] [PanopTILs Refined](https://huggingface.co/datasets/csmp-hub/panoptils_refined) | 1,349 | Patient-separated `train`, `val`, `test` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [[11](#references)] [PUMA](https://huggingface.co/datasets/csmp-hub/puma) | 206 | Public challenge `train` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| [[12](#references)] [CytoNuke](https://huggingface.co/datasets/csmp-hub/cytonuke) | 83 | New patient-separated `train`, `val`, `test` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (CPTAC images: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) |
| [[13](#references)] [HeLaCytoNuc](https://huggingface.co/datasets/csmp-hub/helacytonuke) | 2,676 | Original `train`, `val` (automatic labels), `test` (manual labels) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [[14](#references)] [AIR-LEISH](https://huggingface.co/datasets/csmp-hub/air-leish) | 180 | `train` only; no independent evaluation split | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [[15, 16](#references)] [Lizard-Mitosis](https://huggingface.co/datasets/csmp-hub/lizard-mitosis) | 4,981 | Original `train`, `val`, `test`; train/val overlap | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (noncommercial) |
| [[17](#references)] [Human Lung](https://huggingface.co/datasets/csmp-hub/human-lung) | 68 | Original `train` (51), `val` (7), `test` (10); patient independence unverified | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (noncommercial) |

For example, download the PanopTILs Refined training split with
[`hf_hub_download`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_download)
and read one sample with
[`ParquetFile.iter_batches`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html#pyarrow.parquet.ParquetFile.iter_batches):

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

RGB images use HWC order; masks use HW axes and integer labels. The PNG decoder
also works for PanNuke, PUMA, CytoNuke, HeLaCytoNuc, AIR-LEISH and Lizard-Mitosis.
Human Lung images use CYX order (channel, height, width); its card shows how to
read the four-channel TIFFs. PanopTILs Refined, PUMA and Human Lung have a `sem`
field. Each linked dataset card lists its filenames, class mappings, split
protocol, attribution and plotting examples.
Class numbers and tissue-background conventions are dataset-specific.

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
- [10] Liu, S., Amgad, M., Rathore, M. A., Salgado, R., & Cooper, L. A. (2022). A panoptic segmentation approach for tumor-infiltrating lymphocyte assessment: development of the MuTILs model and PanopTILs dataset. medRxiv. https://doi.org/10.1101/2022.01.08.22268814.
- [11] Schuiveling, M., et al. (2025). A novel dataset for nuclei and tissue segmentation in melanoma with baseline nuclei segmentation and tissue segmentation benchmarks. GigaScience, 14, giaf011. https://doi.org/10.1093/gigascience/giaf011.
- [12] Raufeisen, J., et al. (2024). Cyto R-CNN and CytoNuke Dataset: Towards reliable whole-cell segmentation in bright-field histological images. Computer Methods and Programs in Biomedicine, 252, 108215. https://doi.org/10.1016/j.cmpb.2024.108215.
- [13] De, T.; Urbanski, A.; Thangamani, S.; Wyrzykowska, M.; Yakimovich, A. (2024). HeLaCytoNuc: fluorescence microscopy dataset with segmentation masks for cell nuclei and cytoplasm. Version 1. RODARE. https://doi.org/10.14278/rodare.3001.
- [14] Oualha, R., Fekih-Romdhane, N., Driss, D., Abdelkrim, Y. Z., Guizani, I., & Harigua-Souiai, E. (2026). AIR-LEISH: A Dataset of Giemsa-Stained Microscopy Images for AI-based Leishmania amastigotes Detection. Scientific Data, 13, 328. https://doi.org/10.1038/s41597-026-06676-8.
- [15] Baumann, E., Dislich, B., Rumberger, J. L., Nagtegaal, I. D., Martínez Rodríguez, M., & Zlobec, I. (2024). HoVer-NeXt: A Fast Nuclei Segmentation and Classification Pipeline for Next Generation Histopathology — Datasets. Version 2. Zenodo. https://doi.org/10.5281/zenodo.11657620.
- [16] Graham, S., Jahanifar, M., Azam, A., et al. (2021). Lizard: A Large-Scale Dataset for Colonic Nuclear Instance Segmentation and Classification. ICCV Workshops, pp. 684–693. https://arxiv.org/abs/2108.11195.
- [17] Dohmen, M., Mittermaier, M., & Hocke, A. (2024). Human Lung Tissue Microscopy (DIC, Fluorescence, Cell and Nuclei Semantic Instance Annotations). Version v1. Zenodo. https://doi.org/10.5281/zenodo.10669918.

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

This project is distributed under [MIT License](https://github.com/okunator/cellseg_models.pytorch/blob/main/LICENSE)

The project contains code from the original cell segmentation and 3rd-party libraries that have permissive licenses:

- [timm](https://github.com/rwightman/pytorch-image-models) (Apache-2)
- [HoVer-Net](https://github.com/vqdang/hover_net) (MIT)
- [Cellpose](https://github.com/MouseLand/cellpose) (BSD-3)
- [Stardist](https://github.com/stardist/stardist) (BSD-3)

If you find this library useful in your project, it is your responsibility to ensure you comply with the conditions of any dependent licenses. Please create an issue if you think something is missing regarding the licenses.
