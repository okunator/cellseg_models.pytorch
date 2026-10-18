# Losses

Use losses with raw network outputs during training, before calling `predict`.
The [training guide](../tutorials/training.ipynb) shows how to prepare predictions
and targets.

B is batch size, C channels, H height and W width. Keep predictions and targets on
the same device. Use integer class labels and floating targets for flow or distance
maps.

## Choose a loss

| Loss | Prediction and target | What it computes |
| --- | --- | --- |
| `CELoss` | BCHW logits; BHW class IDs | Mean multiclass cross-entropy |
| `DiceLoss` | BCHW logits; BHW class IDs | Dice loss, averaged over classes present in each target image |
| `FocalLoss` | BCHW logits; BHW class IDs | Cross-entropy that gives more weight to poorly predicted pixels |
| `MSE` | BCHW raw maps; matching floating targets | Mean squared error; useful for predicting flow maps |
| `MAE` | BCHW raw maps; matching floating targets | Mean absolute error; masking changes how errors are averaged |
| `BCELoss` | BCHW maps; binary targets | Current implementation clips predictions before BCE-with-logits |
| `SCELoss` | BCHW logits; BHW class IDs | Can return infinite loss unless target smoothing is enabled |
| `SSIM` | Comparable floating image/maps | Compares local image patterns; can also return a contrast difference |
| `MSSSIM` | Comparable BCHW maps, at least 32 pixels per axis | Compares structure at five scales; differs from standard MS-SSIM |
| `InstanSegCriterion` | B4HW maps; BHW instance IDs | Seed and center-offset losses; call directly with `pred` |

`IoULoss` and `TverskyLoss` need corrections before use and are not covered here.
Check the limitations above before choosing `BCELoss`, `SCELoss` or `MSSSIM`.

## Combine losses for a model output

```python
import torch

from cellseg_models_pytorch.losses import (
    CELoss,
    DiceLoss,
    JointLoss,
    MSE,
    MultiTaskLoss,
)

criterion = MultiTaskLoss({
    "cellpose": JointLoss([MSE()]),
    "type_mask": JointLoss([CELoss(), DiceLoss()]),
})
flow_prediction = torch.zeros(1, 2, 8, 8, requires_grad=True)
class_logits = torch.zeros(1, 3, 8, 8, requires_grad=True)
loss = criterion(
    yhats={"cellpose": flow_prediction, "type_mask": class_logits},
    targets={
        "cellpose": torch.zeros_like(flow_prediction),
        "type_mask": torch.zeros(1, 8, 8, dtype=torch.long),
    },
)
loss.backward()
```

`JointLoss` adds losses for one output; `MultiTaskLoss` adds losses across named
outputs. Weights multiply each loss and need not sum to one. Supply one weight
per loss. With SSIM, use `return_cs=False` so it returns a single tensor.

Prediction and target dictionaries must have the same keys, including every
configured output. If you supply `edgeweight`, include it in both dictionaries.
Call `InstanSegCriterion` directly with `pred`; the composed loss helpers do not
support it.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.JointLoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.MultiTaskLoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.Loss
   :members: forward
```

## Weighting, masking and smoothing

Options vary by loss; check its parameters below.

- `apply_sd` penalizes large prediction values.
- `apply_ls` softens class labels; `apply_svls` also uses neighboring pixels.
- `class_weights` weights classes when targets contain class IDs.
- `edge_weight` scales each pixel's loss by `edge_weight ** target_weight`.
- `apply_mask=True` enables masking. Masked averages differ by loss; see each
  loss's description before comparing their values.

## Multiclass losses

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.CELoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.DiceLoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.FocalLoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.criterions.sce.SCELoss
   :members: forward
```

## Binary loss

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.BCELoss
   :members: forward
```

## Losses for flow and distance maps

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.MSE
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.MAE
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.SSIM
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.MSSSIM
   :members: forward
```

## InstanSeg loss

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.criterions.instanseg.InstanSegCriterion
   :members: forward
```
