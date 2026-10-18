# API reference

Look up how to create a model, prepare its input and interpret its results.
For complete workflows, start with the [quick start](../quickstart.md) or
[user guide](../user-guide.md).

[Model classes](models.md) load weights and predict images. Their `.model`
attribute gives you the PyTorch network for training. Use
[network factories](model-networks.md) to create a network directly.

```{toctree}
:maxdepth: 1

models
model-networks
inference
slide-reading
wsi-inference
export
datasets
losses
metrics
transforms
file-io
```
