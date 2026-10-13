# cellseg-models.pytorch

Build 2D cell and nuclei segmentation workflows with PyTorch models, aligned
training targets and model-specific instance reconstruction.

These pages describe **current main**, whose maintenance changes have not yet
been released. The published PyPI package retains its own requirements and APIs.
Start with CPU execution, then select the hardware and optional packages your
workflow needs. Synthetic examples demonstrate mechanics, not segmentation quality.

```{toctree}
:maxdepth: 2

installation
tutorials/inference
tutorials/training
tutorials/finetuning
releasing
```

The [repository README](https://github.com/okunator/cellseg_models.pytorch#readme)
contains the quick start and citations. The model implementations can differ from
original papers; consult their documentation before assuming architectural parity.
