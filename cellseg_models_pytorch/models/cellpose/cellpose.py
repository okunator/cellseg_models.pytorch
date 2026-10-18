from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei

__all__ = ["CellPose"]


class CellPose(BaseModelInst):
    """Find nuclei and predict their cell types with a CellPose network.

    Use ``model`` to access the PyTorch network for training. For image
    prediction, call ``set_inference_mode`` before ``predict`` and
    ``post_process``. Loading weights alone does not enable inference.

    The architecture follows `CellPose
    <https://www.nature.com/articles/s41592-020-01018-x>`_.
    """

    model_name = "cellpose"

    def __init__(
        self,
        n_nuc_classes: int,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Configure the network and move it to the requested device.

        Args:
            n_nuc_classes: Number of cell-type output channels, including any
                background class in your label mapping.
            enc_name: Encoder name supported by timm (PyTorch Image Models).
            enc_pretrain: Load pretrained encoder weights. This may download
                weights; it does not load trained segmentation layers.
            enc_freeze: Disable gradient updates to the encoder weights.
                Training mode still controls its normalization layers.
            device: Device on which to create the network. Pass
                ``torch.device("cpu")`` for CPU use; the default requires CUDA.
            model_kwargs: Additional options for ``CellPoseUnet``, such as
                decoder settings. These must match the architecture when loading
                an existing checkpoint.

        Example:
            Create an untrained network without downloading encoder weights:

            >>> import torch
            >>> from cellseg_models_pytorch.models.cellpose import CellPose
            >>> model = CellPose(
            ...     3,
            ...     enc_name="resnet18",
            ...     enc_pretrain=False,
            ...     device=torch.device("cpu"),
            ... )
            >>> model.inference_mode
            False
        """
        super().__init__()
        self.model = cellpose_nuclei(
            n_nuc_classes,
            enc_name=enc_name,
            enc_pretrain=enc_pretrain,
            enc_freeze=enc_freeze,
            **model_kwargs,
        )

        self.device = device
        self.model.to(device)

    def set_inference_mode(self, mixed_precision: bool = True) -> None:
        """Prepare image prediction and instance reconstruction.

        Sets the network to evaluation mode and replaces its predictor and
        post-processor. This affects later training; call ``model.train()`` on the
        underlying network when returning to training.

        CellPose reconstruction initially uses ``use_gpu=True``. For CPU-only
        reconstruction, replace ``post_processor`` with
        ``PostProcessor("cellpose", {"use_gpu": False})`` after this call.

        Args:
            mixed_precision: Use float16 autocasting during prediction. Choose False
                for ordinary float32 inference, especially on CPU. Reconstruction
                has its own device settings.
        """
        self.model.eval()
        self.predictor = Predictor(
            model=self.model,
            mixed_precision=mixed_precision,
        )
        self.post_processor = PostProcessor(
            postproc_method="cellpose",
            postproc_kwargs={"use_gpu": True},
        )
        self.inference_mode = True

    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        opset_version: int = 18,
        dynamic_batch: bool = True,
    ) -> Path:
        """Save this network's dense prediction stage as an ONNX file.

        Args:
            output_path: Destination .onnx path. Parent folders are created and an
                existing file may be overwritten.
            input_shape: Example (batch, channels, height, width) shape. Spatial
                dimensions are fixed in the exported graph.
            opset_version: ONNX operator-set version supported by your deployment runtime.
            dynamic_batch: Allow the batch dimension to vary at inference time.

        Returns:
            Path to the exported model. Python instance reconstruction is not included.

        See Also:
            ``export_cellpose_onnx`` for dependencies, output names and state restoration.
        """
        from cellseg_models_pytorch.models.cellpose.onnx import export_cellpose_onnx

        return export_cellpose_onnx(
            self,
            output_path=output_path,
            input_shape=input_shape,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
