from typing import Any, Dict

import torch

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.inference.predictor import Predictor
from cellseg_models_pytorch.models.base._base_model_inst import BaseModelInst
from cellseg_models_pytorch.models.cppnet.cppnet_unet import cppnet_nuclei

__all__ = ["CPPNet"]


class CPPNet(BaseModelInst):
    """Find nuclei and predict their cell types with a CPPNet network.

    Use ``model`` for training. Call ``set_inference_mode`` before ``predict``
    and ``post_process`` to enable image inference.
    """

    model_name = "cppnet"

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

        - https://arxiv.org/abs/2102.06867

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
            model_kwargs: Additional ``CPPNetUnet`` options. Use the same architecture
                settings as the checkpoint when loading existing weights.
        """
        super().__init__()
        self.model = cppnet_nuclei(
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

        Configures StarDist reconstruction with bounding-box trimming enabled.

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
            postproc_kwargs={"trim_bboxes": True},
        )
        self.inference_mode = True
