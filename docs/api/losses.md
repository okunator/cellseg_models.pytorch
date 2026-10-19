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
| `IoULoss` | BCHW logits; BHW class IDs | Soft IoU loss, averaged over target-present image/class pairs |
| `TverskyLoss` | BCHW logits; BHW class IDs | Spatial overlap with separate false-positive and false-negative coefficients |
| `FocalLoss` | BCHW logits; BHW class IDs | Cross-entropy that gives more weight to poorly predicted pixels |
| `MSE` | BCHW raw maps; matching floating targets | Mean squared error; useful for predicting flow maps |
| `MAE` | BCHW raw maps; matching floating targets | Mean absolute error; masking changes how errors are averaged |
| `BCELoss` | BCHW logits; BCHW binary/soft targets, or a shared BHW binary map | Independent binary cross-entropy per output channel |
| `SCELoss` | BCHW logits; BHW class IDs | Weighted forward and reverse multiclass cross-entropy |
| `SSIM` | Comparable floating image/maps | Compares local image patterns; can also return a contrast difference |
| `MSSSIM` | Matching floating BCHW maps, at least 161 pixels per axis with the default window | 1 minus five-scale MS-SSIM, averaged over images and channels |
| `InstanSegCriterion` | B4HW maps; BHW instance IDs | Seed BCE and per-nucleus membership loss, using the model's pixel classifier |

For `SSIM`, set `data_range=1` for normalized maps or `data_range=255` for
floating images in [0, 255]. The default `None` retains prediction-only range
guessing and can change with batch contents. Floating BHW targets are shared
across channels; integer BHW targets are class IDs converted to one-hot maps.
The loss retains its cap at 0.5, so negative similarities have zero gradient.

For `MSSSIM`, supply maps in [0, 1], or set `data_range=255` for floating images
in [0, 255]. It does not convert logits to probabilities. Masking, weighting and
smoothing options are unused.

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
InstanSeg supports these helpers when given the network's pixel classifier;
see [InstanSeg loss](#instanseg-loss).

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
- `class_weights` weights classes when targets contain class IDs; for BCE it
  weights independent output channels.
- `edge_weight` sets per-pixel weights as `edge_weight ** target_weight`; how
  these enter the calculation depends on the loss.
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
.. autoclass:: cellseg_models_pytorch.losses.IoULoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.TverskyLoss
   :members: forward
```

Tversky uses `TP / (TP + alpha * FP + beta * FN)` for each image and class,
then averages the losses for target-present classes, including background.
`alpha` weights false positives and `beta` weights false negatives. Using
`alpha=beta=0.5` gives Dice loss; `alpha=beta=1` gives IoU loss.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.FocalLoss
   :members: forward
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.criterions.sce.SCELoss
   :members: forward
```

SCE floors target probabilities at `1e-4` only inside the reverse logarithm,
as in the [authors' implementation](https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/blob/master/loss.py).
This keeps hard-label reverse terms finite without smoothing the forward term.

## Binary loss

Pass raw logits. Each channel is an independent binary task, and targets must be
in `[0, 1]`. `class_weights` supplies one multiplier per channel, including when
there is only one channel; it does not select background/foreground weights.
Uniform label smoothing moves binary targets toward `0.5`.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.BCELoss
   :members: forward
```

## Losses for flow and distance maps

MSE and MAE compute half-precision inputs in float32. MSE masking retains the
full-image mean; MAE divides by the mean mask weight and can also penalize
predictions outside the mask. Regularization is added after masking, so a fully
masked input can still contribute a penalty.

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

(instanseg-loss)=
## InstanSeg loss

Pass the network's existing `probability_net` so training and reconstruction use
the same learned classifier. Optimize the network parameters, which include it:

```python
from cellseg_models_pytorch.losses import Loss
from cellseg_models_pytorch.models.instanseg import instanseg_nuclei

network = instanseg_nuclei(3, enc_name="resnet18", enc_pretrain=False)
instance_criterion = Loss(
    "instanseg", pixel_classifier=network.probability_net,
)
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
```

Apply this criterion to `network(images)["nuc"].aux_map` and BHW **instance IDs**,
not cell-type labels. Train the cell-type head separately with a classification
loss. Direct calls accept `pred` or `yhat`; `JointLoss` and `MultiTaskLoss` use
`yhat` automatically.

The objective follows [upstream InstanSeg](https://github.com/instanseg/instanseg/blob/414fbc9e8039c1d049cc27871d66409084ded21c/instanseg/utils/loss/instanseg_loss.py):
seed binary cross-entropy plus per-crop Lovász hinge for nucleus membership.
Match `window_size` and `peak_distance` to reconstruction. At most 50 seed crops
per image are sampled by default. Images without a foreground seed above 0.5
have only seed loss. Negative labels mark unannotated pixels; an optional binary
`mask` excludes additional pixels from both losses and seed selection. Fully
excluded images contribute zero to the batch average.

Generic smoothing, spectral decoupling, class weights and edge weights are
unsupported. The mask is applied when supplied, without `apply_mask=True`.
The old coordinate MSE is replaced by membership training; `w_coord` is a
deprecated alias for `w_inst`. Reconstruction thresholds seed **probabilities**
after sigmoid, while the input maps remain raw logits. Existing checkpoint
shapes and parameter names are unchanged; the corrected objective requires
retraining models trained only with the old criterion.

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.losses.criterions.instanseg.InstanSegCriterion
   :members: forward
```
