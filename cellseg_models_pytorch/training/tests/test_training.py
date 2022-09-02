from copy import deepcopy

import pytest
import pytorch_lightning as pl

from cellseg_models_pytorch.datamodules.custom_datamodule import CustomDataModule
from cellseg_models_pytorch.datasets import SegmentationFolderDataset
from cellseg_models_pytorch.models import cellpose_plus
from cellseg_models_pytorch.training.lit import SegmentationExperiment


# @pytest.mark.parametrize
def test_training(img_patch_dir, mask_patch_dir):
    train_ds = SegmentationFolderDataset(
        path=img_patch_dir.as_posix(),
        mask_path=mask_patch_dir.as_posix(),
        img_transforms=["blur"],
        inst_transforms=["cellpose"],
        return_sem=True,
        return_type=True,
        return_inst=False,
        return_weight=False,
        normalization="percentile",
    )
    valid_ds = deepcopy(train_ds)
    test_ds = deepcopy(train_ds)

    datamodule = CustomDataModule(
        [train_ds, valid_ds, test_ds], batch_size=1, num_workers=1
    )

    model = cellpose_plus(sem_classes=7, type_classes=7)
    experiment = SegmentationExperiment(
        model=model,
        branch_losses={"cellpose": "mse_ssim", "sem": "ce_dice", "type": "ce_dice"},
        branch_metrics={"cellpose": [None], "sem": ["miou"], "type": ["miou"]},
        lookahead=False,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=0,
        profiler="simple",
        move_metrics_to_cpu=True,
        fast_dev_run=True,
    )

    trainer.fit(model=experiment, datamodule=datamodule)
