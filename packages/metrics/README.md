# Cell segmentation metrics

Binary, multiclass and instance segmentation metrics using NumPy and SciPy.
PyTorch and slide-reading packages are not required.

From a source checkout, install this package independently:

```sh
python -m pip install ./packages/metrics
```

Compare integer instance masks, where zero is background and positive values
identify separate nuclei:

```python
import numpy as np

from cellseg_metrics import panoptic_quality

reference = np.array([[0, 1, 1, 0, 2, 2]])
prediction = np.array([[0, 7, 7, 0, 0, 0]])

scores = panoptic_quality(reference, prediction)
print(scores)
```

`pq` measures instance segmentation quality, `dq` measures detection quality,
and `sq` measures overlap quality for matched instances. Instance IDs do not
need to match between the masks.

The full `cellseg-models-pytorch` library preserves its existing
`cellseg_models_pytorch.metrics` imports and uses this same implementation.
See the [metrics reference](https://okunator.github.io/cellseg_models.pytorch/api/metrics.html)
for inputs, outputs and empty-mask behavior.
