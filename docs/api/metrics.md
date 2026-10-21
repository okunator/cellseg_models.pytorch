# Metrics

Compare predicted masks with reference labels. Use NumPy arrays on CPU:
instance masks for individual cells, class masks for categories, or binary masks
for foreground/background. Convert raw model scores to labels first.

Install the standalone metrics package without PyTorch:

```shell
pip install cellseg-metrics
```

It requires only NumPy and SciPy. The full library preserves its existing
metric imports through `cellseg_models_pytorch.metrics`; import new functions
directly from `cellseg_metrics`.

## Instance evaluation

Use 2D integer masks with 0 for background and a positive ID for each object.
IDs can differ between prediction and reference masks. Panoptic quality matches
objects by their overlap and returns `pq`, `sq` and `dq`.

```python
import numpy as np

from cellseg_metrics import panoptic_quality

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
| `precision_recall_ap` | Area under the confidence-ranked precision–recall envelope at one IoU threshold |
| `mean_average_precision` | COCO-style 101-point AP averaged over IoU thresholds, by default 0.50–0.95 |
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

## Confidence-ranked average precision

Use `precision_recall_ap` when each predicted nucleus has a confidence score.
Pass labelled instance masks and a mapping from prediction ID to confidence.
Higher scores mean more confident detections; instance IDs themselves are not scores.

```python
from cellseg_metrics import mean_average_precision, precision_recall_ap

confidences = {12: 0.9}
ap50 = precision_recall_ap(truth, prediction, confidences, iou_threshold=0.5)
mean_ap = mean_average_precision(truth, prediction, confidences)
```

Predictions are considered from highest to lowest confidence. Each can match one
unmatched reference nucleus with IoU at least the threshold. Missed nuclei reduce
recall; confident false detections reduce precision. `ap50` integrates the full
interpolated precision–recall curve. `mean_ap` instead samples its precision envelope
at 101 recall levels and averages across IoU 0.50, 0.55, ..., 0.95, following the
[COCO evaluation convention](https://cocodataset.org/dataset/detection-eval.htm).
Both scores are 1 for the matching masks above.

For a dataset, pass sequences of reference masks, prediction masks and confidence
mappings. Scores are ranked across all images, while objects only match within
their own image. Do not average per-image AP to obtain dataset AP. Evaluate each
cell type separately; these functions do not average across classes or apply
COCO's crowd labels, area ranges or 100-detection limit.

Both functions return zero when reference nuclei exist but none are detected.
They return `NaN` when the entire input has no reference nuclei: AP is undefined
in that case. The historical `average_precision` remains object precision at one
IoU threshold and retains its previous behavior.

```{eval-rst}
.. autofunction:: cellseg_metrics.precision_recall_ap
```

```{eval-rst}
.. autofunction:: cellseg_metrics.mean_average_precision
```

## Binary pixel evaluation

Pass 0/1 masks to `get_stats`, then pass its three result arrays to a score
function. Keep these arrays per pixel when using `accuracy`.

```python
from cellseg_metrics import get_stats, iou_score

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
