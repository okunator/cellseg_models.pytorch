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

These examples use the source version; the PyPI release may have different
requirements and APIs. Source installations are tested on
Python 3.10–3.12. Python 3.14.1 is excluded by torchvision; other newer versions
have not yet been validated here. For an NVIDIA GPU, follow
[PyTorch's installation guidance](https://pytorch.org/get-started/locally/)
for your hardware. Examples use CUDA when available and otherwise use CPU.
