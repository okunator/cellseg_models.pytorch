# Metrics

Compare predicted masks with reference labels. Use NumPy arrays on CPU:
instance masks for individual cells, class masks for categories, or binary masks
for foreground/background. Convert raw model scores to labels first.

These functions also belong to the standalone `cellseg-metrics` package, which
requires only NumPy and SciPy. For example, use
`from cellseg_metrics import panoptic_quality` without installing PyTorch.
See the [source installation instructions](https://github.com/okunator/cellseg_models.pytorch#metrics-only)
until its first PyPI release.

The full library depends on `cellseg-metrics`. Its existing
`cellseg_models_pytorch.metrics` imports remain available and re-export the same
functions, so the examples below work with either import path.

## Instance evaluation

Use 2D integer masks with 0 for background and a positive ID for each object.
IDs can differ between prediction and reference masks. Panoptic quality matches
objects by their overlap and returns `pq`, `sq` and `dq`.

```python
import numpy as np

from cellseg_models_pytorch.metrics import panoptic_quality

truth = np.array([[0, 5, 5], [0, 0, 0]], dtype=np.int32)
prediction = np.array([[0, 12, 12], [0, 0, 0]], dtype=np.int32)
score = panoptic_quality(truth, prediction)
```

`score["pq"]` is nearly 1 because the object regions match. These instance scores
return zero when either mask is empty, including when both are empty. Each call
evaluates one image pair.

| Function | Meaning and matching |
| --- | --- |
| `panoptic_quality` | `pq = dq * sq`; `dq` measures missed and extra objects (F1); `sq` measures how closely matched objects overlap (IoU) |
| `average_precision` | Fraction of predicted objects matched at one overlap threshold; not average precision across a curve |
| `aggregated_jaccard_index` | Best-overlapping prediction per reference object; matches can be reused and `thresh` is ignored |
| `dice2` | Dice over overlapping object pairs; ignores pairs with no overlap |

For scores by cell type, evaluate instance masks separately for each class.
See the [evaluation guide](../tutorials/evaluation.ipynb) for an example.

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.panoptic_quality
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.average_precision
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.aggregated_jaccard_index
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.dice2
```

## Binary pixel evaluation

Pass 0/1 masks to `get_stats`, then pass its three result arrays to a score
function. Keep these arrays per pixel when using `accuracy`.

```python
from cellseg_models_pytorch.metrics import get_stats, iou_score

binary_truth = (truth > 0).astype(np.uint8)
binary_prediction = (prediction > 0).astype(np.uint8)
tp, fp, fn = get_stats(binary_truth, binary_prediction)
foreground_iou = iou_score(tp, fp, fn)
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.get_stats
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.iou_score
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.dice_coef
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.accuracy
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.f1score
```

## Class-mask evaluation

Use 2D class masks with labels from 0 to `num_classes - 1`. Results contain one
score per class, including class 0. With `clamp_absent=True`, a class absent
from both masks gets -1; leave those values out of averages. Classes present in
only one mask are still evaluated.

`specificity_multiclass` measures how often pixels outside a reference class
are correctly excluded from that predicted class: TN / (TN + FP).

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.iou_multiclass
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.dice_multiclass
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.f1score_multiclass
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.accuracy_multiclass
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.sensitivity_multiclass
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.specificity_multiclass
```

## Pairwise statistics

Use these helpers for counts between object or class pairs. Class-mode pixel
counts include missed and extra pixels even when a class has no correct overlap.
Instance mode covers only overlapping object pairs. Object counts need a match
matrix with at most one match per object; the helper does not choose matches.

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.pairwise_pixel_stats
```

```{eval-rst}
.. autofunction:: cellseg_models_pytorch.metrics.pairwise_object_stats
```
