# Installation

## From PyPI

Install the latest published release in your Python environment:

```sh
python -m pip install cellseg-models-pytorch
```

### Optional features

Install the features you need:

| Feature | Command | Includes |
| --- | --- | --- |
| Training datasets and augmentation | `python -m pip install "cellseg-models-pytorch[training]"` | Albumentations and PyTables |
| ONNX export | `python -m pip install "cellseg-models-pytorch[onnx]"` | ONNX and ONNX Script |
| OpenSlide reading | `python -m pip install "cellseg-models-pytorch[openslide]"` | OpenSlide Python and native libraries |

Combine features with commas, for example `"cellseg-models-pytorch[training,onnx]"`.
Install ONNX Runtime separately to run exported models. cuCIM and BioIO require
their own setup.

## From source with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then download
the repository and install its dependencies:

```sh
git clone https://github.com/okunator/cellseg_models.pytorch.git
cd cellseg_models.pytorch
uv sync --locked
```

Run Python from this environment with `uv run python`.
Continue with the [quick start](quickstart.md).

Add optional features with `uv sync --locked --extra training`, using the same
`training`, `onnx` and `openslide` names as above. Combine flags to keep several
features, for example `uv sync --locked --extra training --extra onnx`.
Running `uv sync` without these flags removes the optional packages.

These pages describe version 0.2.0. Installations are tested on
Python 3.10–3.14. Python 3.14.1 is excluded by torchvision. For an NVIDIA GPU, follow
[PyTorch's installation guidance](https://pytorch.org/get-started/locally/)
for your hardware. Examples use CUDA when available and otherwise use CPU.
