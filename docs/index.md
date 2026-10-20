---
html_theme.sidebar_secondary.remove: true
---

# Cell segmentation with PyTorch

```{container} csmp-intro
Find individual cells and nuclei in your images. Use pretrained models,
train on your own labels, and run predictions on images or whole slides.
```

```{container} csmp-actions
[Get started](quickstart.md)
[Installation](installation.md)
[API reference](api/index.md)
```

```sh
pip install cellseg-models-pytorch
```

````{container} csmp-example
```{figure} _static/segmentation-example.png
:alt: A tissue image alongside predicted nuclei and predicted cell types. Different colours distinguish individual nuclei and cell categories.

From the [image inference guide](tutorials/inference.ipynb): predictions on a
PanopTILs Refined validation image. Colours distinguish nuclei and cell types.
```
````

## What would you like to do?

````{container} csmp-paths
```{container} csmp-path
### Predict an image

Load pretrained weights and turn an image into labelled nuclei.

[Follow the quick start](quickstart.md)
```

```{container} csmp-path
### Train on your data

Prepare labelled images, train a model and check its predictions.

[Start with data preparation](tutorials/prepare-data.ipynb)
```

```{container} csmp-path
### Work with whole slides

Read tissue regions, predict tiles and combine the results.

[Explore slide inference](tutorials/wsi.ipynb)
```
````

## Models and tools

Choose from **CellPose, StarDist, Omnipose, HoverNet, CellVit and InstanSeg**.
The [model reference](api/models.md) covers available checkpoints and how to use them.

Use the [user guide](user-guide.md) for complete workflows, or look up
[datasets](api/datasets.md), [transforms](api/transforms.md),
[losses](api/losses.md), [metrics](api/metrics.md) and [ONNX export](api/export.md)
in the API reference.

```{container} csmp-resources
**Find weights and data** — Browse [CSMP Hub](https://huggingface.co/csmp-hub).

**Go further** — For panoptic segmentation and downstream spatial analysis,
see [Histolytics](https://hautaniemilab.github.io/histolytics/), which uses these models.

**Contribute or cite** — Visit the [repository](https://github.com/okunator/cellseg_models.pytorch)
for source code, issues and model citations.
```

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
user-guide
api/index
```
