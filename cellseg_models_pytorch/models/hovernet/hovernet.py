from pathlib import Path
from typing import Any, Dict

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.hovernet.hovernet_unet import hovernet_nuclei

__all__ = ["HoverNet"]


class HoverNet(BaseModelInst):
    """Find nuclei and predict their cell types with a HoverNet network.

    Use ``model`` for training. Call ``set_inference_mode`` before ``predict``
    and ``post_process`` to enable image inference.
    """

    model_name = "hovernet"

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

        - https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub

        Args:
            n_nuc_classes: Number of cell-type output channels, including any background
                class in the training label mapping.
            enc_name: Encoder name supported by timm (PyTorch Image Models).
            enc_pretrain: Load pretrained encoder weights, downloading if needed.
                Segmentation layers still need training or a segmentation checkpoint.
            enc_freeze: Disable encoder weight updates. Training mode still controls
                its normalization layers.
            device: Network device. Pass ``torch.device("cpu")`` for CPU use;
                the default requires CUDA.
            model_kwargs: Additional ``HoverNetUnet`` options. Use the same architecture
                settings as the checkpoint when loading existing weights.
        """
        super().__init__()
        self.model = hovernet_nuclei(
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

        Configures HoVer-Net reconstruction for the horizontal/vertical maps.

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
        self.post_processor = PostProcessor(postproc_method="hovernet")
        self.inference_mode = True

    def export_onnx(
        self,
        output_path: str | Path,
        input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
        opset_version: int = 18,
        dynamic_batch: bool = True,
    ) -> Path:
        """Save prediction maps for use in an ONNX runtime.

        Args:
            output_path: Destination .onnx file; an existing file may be overwritten.
            input_shape: Example (batch, channels, height, width) shape. Image
                height and width stay fixed in the exported file.
            opset_version: ONNX operator-set version, default 18.
            dynamic_batch: Allow the number of images in a batch to vary.

        Returns:
            Exported file path. Instance reconstruction remains in Python.

        See Also:
            ``export_hovernet_onnx`` for output names and optional dependencies.
        """
        from cellseg_models_pytorch.models.hovernet.onnx import export_hovernet_onnx

        return export_hovernet_onnx(
            self, output_path, input_shape, opset_version, dynamic_batch
        )
