# Installation

## From PyPI

Install the latest published release in your Python environment:

```sh
python -m pip install cellseg-models-pytorch
```

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

### Optional features

From the source checkout, install the features you need:

| Feature | Command | Includes |
| --- | --- | --- |
| Training datasets and augmentation | `uv sync --locked --extra training` | Albumentations and PyTables |
| ONNX export | `uv sync --locked --extra onnx` | ONNX and ONNX Script |
| OpenSlide reading | `uv sync --locked --extra openslide` | OpenSlide Python and native libraries |

Combine flags to keep several features, for example
`uv sync --locked --extra training --extra onnx`. Running `uv sync` without
these flags removes the optional packages. ONNX Runtime is installed separately
if you want to run exported files. cuCIM and BioIO still require their own setup.
These extras are new in the source version and are not in the current PyPI release.

These examples use the source version; the PyPI release may have different
requirements and APIs. Source installations are tested on
Python 3.10–3.14. Python 3.14.1 is excluded by torchvision. For an NVIDIA GPU, follow
[PyTorch's installation guidance](https://pytorch.org/get-started/locally/)
for your hardware. Examples use CUDA when available and otherwise use CPU.
