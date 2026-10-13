# Installation

## Current main

Use Python 3.10–3.12, the versions covered by source and clean-install CI.
The maintenance version remains 0.1.30 and is not published yet. From a checkout:

```sh
python -m pip install uv==0.11.4
uv sync --locked --group docs
```

This installs the library, developer dependencies and the separate notebook/docs
group. Runtime requirements include PyTorch >=2.13, NumPy >=2.2.6 and timm >=1.0.30;
Python 3.14.1 is excluded by torchvision. Newer Python versions are not validated.
For the published package, `pip install cellseg-models-pytorch` installs that
release's requirements; do not assume these unreleased examples match it.

For GPU execution, follow [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for the matching GPU/driver build first. CPU validation does not establish GPU/AMP
compatibility. The notebooks use CPU explicitly and disable pretrained encoder
downloads. Real pretrained inference requires an explicitly supplied local checkpoint.

## Notebooks and pages

The tutorials are `.ipynb` files under `docs/tutorials/`. Open them with your
Jupyter client using a kernel from the synchronized environment. They contain
plain Python cells and are the source of the corresponding documentation pages.

Build pages without executing notebooks or downloading weights:

```sh
uv run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Validate all three offline notebooks during the build:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --locked --group docs sphinx-build -W --keep-going -b html -D nb_execution_mode=force docs docs/_build/executed
```

Open `docs/_build/html/index.html` to read the generated site. Build outputs stay
out of Git. Notebook outputs are cleared in source control; execution errors fail
validation. These checks do not train a useful model or validate a real dataset.
