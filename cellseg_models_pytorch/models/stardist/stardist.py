from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_nuclei

__all__ = ["StarDist"]


class StarDist(BaseModelInst):
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
        """Stardist model for nuclei segmentation.

        Stardist:
        - https://arxiv.org/abs/1806.03535

        Parameters:
            n_nuc_classes (int):
                Number of nuclei type classes.
            n_rays (int, default=32):
                Number of predicted rays.
            enc_name (str, default="efficientnet_b5"):
                Name of the pytorch-image-models encoder.
            enc_pretrain (bool, default=True):
                Whether to use pretrained weights in the encoder.
            enc_freeze (bool, default=False):
                Freeze encoder weights for training.
            device (torch.device, default=torch.device("cuda")):
                Device to run the model on. Default is "cuda".
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
        """Set to inference mode."""
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
        """Export the StarDist dense prediction network to ONNX.

        Instance reconstruction and NMS are intentionally left in the existing
        Python post-processing pipeline.
        """
        from cellseg_models_pytorch.models.stardist.onnx import export_stardist_onnx

        return export_stardist_onnx(
            self,
            output_path=output_path,
            input_shape=input_shape,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
        )
