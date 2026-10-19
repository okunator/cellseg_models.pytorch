# Models

Load weights and predict individual cells with a model class. For training,
use its `.model` PyTorch network. Start with the [quick start](../quickstart.md)
or [image inference guide](../tutorials/inference.ipynb).

## Choose a family and checkpoint

| Family | HGSC checkpoint name | Cervix (CIN2) |
| --- | --- | --- |
| CellPose | `hgsc_v1_efficientnet_b5` | `cin2_v1_efficientnet_b5` |
| StarDist | `hgsc_v1_efficientnet_b5` | `cin2_v1_efficientnet_b5` |
| HoverNet | `hgsc_v1_efficientnet_b5` | `cin2_v1_efficientnet_b5` |
| CPPNet | `hgsc_v1_efficientnet_b5` | `cin2_v1_efficientnet_b5` |
| CellVit | `hgsc_v1_efficientnet_b5` | `cin2_v1_samvit_base_patch16` |
| InstanSeg | Use local weights | — |

Load pretrained weights by checkpoint name, or pass a local path to load your own.
Named checkpoints may download weights from Hugging Face. Use the normalization
and class labels supplied with the weights.

CIN2 repositories currently contain model cards only; weights cannot yet be
loaded by name. Load any compatible local CIN2 weights by path.

## Predict images

Call `set_inference_mode()` before prediction. For CPU prediction, pass
`mixed_precision=False`. CellPose also needs CPU reconstruction configured as
shown in the [image inference guide](../tutorials/inference.ipynb).

`predict` produces maps; `post_process` turns them into instance and class masks.
See [inference](inference.md) for input formats and output fields.

## Train a model

Use `.model` with a normal PyTorch training loop. After training, call
`set_inference_mode()` again before predicting images. The
[training guide](../tutorials/training.ipynb) includes encoder freezing; the
[network reference](model-networks.md) covers architecture options.

## CellPose

```python
from cellseg_models_pytorch.models.cellpose import CellPose
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.cellpose.CellPose
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellpose.CellPose.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellpose.CellPose.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellpose.CellPose.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.cellpose.CellPose.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.cellpose.CellPose.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.cellpose.CellPose.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.cellpose.CellPose.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

ONNX export: {py:meth}`~cellseg_models_pytorch.models.cellpose.CellPose.export_onnx` in the [export reference](export.md).

## StarDist

`from_pretrained` reads the ray count from the saved weights.

```python
from cellseg_models_pytorch.models.stardist import StarDist
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.stardist.StarDist
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.stardist.StarDist.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.stardist.StarDist.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.stardist.StarDist.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.stardist.StarDist.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.stardist.StarDist.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.stardist.StarDist.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.stardist.StarDist.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

ONNX export: {py:meth}`~cellseg_models_pytorch.models.stardist.StarDist.export_onnx` in the [export reference](export.md).

## HoverNet

```python
from cellseg_models_pytorch.models.hovernet import HoverNet
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.hovernet.HoverNet
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.hovernet.HoverNet.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.hovernet.HoverNet.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.hovernet.HoverNet.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.hovernet.HoverNet.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.hovernet.HoverNet.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.hovernet.HoverNet.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.hovernet.HoverNet.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

ONNX export: {py:meth}`~cellseg_models_pytorch.models.hovernet.HoverNet.export_onnx` in the [export reference](export.md).

## CPPNet

`from_pretrained` reads the ray count from the saved weights.

```python
from cellseg_models_pytorch.models.cppnet import CPPNet
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.cppnet.CPPNet
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cppnet.CPPNet.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cppnet.CPPNet.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cppnet.CPPNet.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.cppnet.CPPNet.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.cppnet.CPPNet.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.cppnet.CPPNet.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.cppnet.CPPNet.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

## CellVit

ONNX export: {py:meth}`~cellseg_models_pytorch.models.cellvit.CellVit.export_onnx` in the [export reference](export.md).

Use a supported `samvit_*` encoder. Loading a checkpoint selects the encoder
stored in its weights, which can differ from the constructor default.

```python
from cellseg_models_pytorch.models.cellvit import CellVit
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.cellvit.CellVit
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellvit.CellVit.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellvit.CellVit.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.cellvit.CellVit.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.cellvit.CellVit.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.cellvit.CellVit.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.cellvit.CellVit.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.cellvit.CellVit.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

## InstanSeg

```python
from cellseg_models_pytorch.models.instanseg import InstanSeg
```

```{eval-rst}
.. autoclass:: cellseg_models_pytorch.models.instanseg.InstanSeg
   :members: set_inference_mode
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.instanseg.InstanSeg.from_pretrained
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.instanseg.InstanSeg.predict
```

```{eval-rst}
.. automethod:: cellseg_models_pytorch.models.instanseg.InstanSeg.post_process
```

```{eval-rst}
.. py:attribute:: cellseg_models_pytorch.models.instanseg.InstanSeg.model

   PyTorch network used for training.

.. py:attribute:: cellseg_models_pytorch.models.instanseg.InstanSeg.device

   Device used by the model.

.. py:attribute:: cellseg_models_pytorch.models.instanseg.InstanSeg.inference_mode

   Whether inference has been set up. Calling ``.model.train()`` does not reset this flag.

.. py:attribute:: cellseg_models_pytorch.models.instanseg.InstanSeg.post_processor

   Post-processor created by ``set_inference_mode``. Replace it to change reconstruction settings.

```

ONNX export: {py:meth}`~cellseg_models_pytorch.models.instanseg.InstanSeg.export_onnx` in the [export reference](export.md).
