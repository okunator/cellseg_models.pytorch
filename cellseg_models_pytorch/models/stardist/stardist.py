from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_nuclei

__all__ = ["StarDist"]


class StarDist(BaseModelInst):
    """Find nuclei and predict their cell types with a StarDist network.

    Use ``model`` for training. Call ``set_inference_mode`` before ``predict``
    and ``post_process`` to enable image inference.
    """

    model_name = "stardist"

    def __init__(
        self,
        n_nuc_classes: int,
        n_rays: int = 32,
        enc_name: str = "efficientnet_b5",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Configure the network and move it to the requested device.

        - https://arxiv.org/abs/1806.03535

        Args:
            n_nuc_classes: Number of cell-type output channels, including any background
                class in the training label mapping.
            n_rays: Number of radial distances per pixel. Match the training targets
                and checkpoint; loading weights does not infer this setting.
            enc_name: Encoder name supported by timm (PyTorch Image Models).
            enc_pretrain: Load pretrained encoder weights, downloading if needed.
                Segmentation layers still need training or a segmentation checkpoint.
            enc_freeze: Disable encoder weight updates. Training mode still controls
                its normalization layers.
            device: Network device. Pass ``torch.device("cpu")`` for CPU use;
                the default requires CUDA.
            model_kwargs: Additional ``StarDistUnet`` options. Use the same architecture
                settings as the checkpoint when loading existing weights.
        """
        super().__init__()
        self.model = stardist_nuclei(
            n_rays=n_rays,
            n_nuc_classes=n_nuc_classes,
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

        Configures StarDist reconstruction with bounding-box trimming and
        normalization enabled.

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
            postproc_method="stardist",
            postproc_kwargs={"trim_bboxes": True, "normalize": True},
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
            ``export_stardist_onnx`` for dependencies, output names and state restoration.
        """
        from cellseg_models_pytorch.models.stardist.onnx import export_stardist_onnx

        return export_stardist_onnx(
            self,
            output_path=output_path,
            input_shape=input_shape,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
