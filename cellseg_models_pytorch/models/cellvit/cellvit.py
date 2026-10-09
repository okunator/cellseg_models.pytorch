from typing import Any, Dict

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.cellvit.cellvit_unet import cellvit_nuclei

__all__ = ["CellVit"]


class CellVit(BaseModelInst):
    model_name = "cellvit"

    def __init__(
        self,
        n_nuc_classes: int,
        enc_name: str = "samvit_base_patch16",
        enc_pretrain: bool = True,
        enc_freeze: bool = False,
        device: torch.device = torch.device("cuda"),
        model_kwargs: Dict[str, Any] = {},
    ) -> None:
        """CellViT model for nuclei segmentation.

        CellVit:
        - https://arxiv.org/abs/2306.15350

        Args:
            n_nuc_classes: Number of nuclei type classes.
            enc_name: Name of the pytorch-image-models encoder.
                Defaults to "samvit_base_patch16".
            enc_pretrain: Whether to use pretrained weights in the encoder.
                Defaults to True.
            enc_freeze: Freeze encoder weights for training.
                Defaults to False.
            device: Device to run the model on. Default is "cuda".
                Defaults to torch.device("cuda").
        """
        super().__init__()
        self.model = cellvit_nuclei(
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
        self.post_processor = PostProcessor(postproc_method="hovernet")
        self.inference_mode = True
