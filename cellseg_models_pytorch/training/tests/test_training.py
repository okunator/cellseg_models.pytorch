import pytest
import pytorch_lightning as pl

from cellseg_models_pytorch.datamodules.csmp_datamodule import CSMPDataModule
from cellseg_models_pytorch.models import cellpose_plus
from cellseg_models_pytorch.training import SegmentationExperiment


# @pytest.mark.parametrize
def test_training(img_patch_dir, mask_patch_dir):
    type_classes = {
        "background": 0,
        "class1": 1,
        "class2": 2,
        "class3": 3,
        "class4": 4,
        "class5": 5,
        "class6": 6,
    }

    sem_classes = {
        "background": 0,
        "class1": 1,
        "class2": 2,
        "class3": 3,
        "class4": 4,
        "class5": 5,
        "class6": 6,
    }

    model = cellpose_plus(sem_classes=7, type_classes=7)

    datamodule = CSMPDataModule(
        img_transforms=["blur"],
        inst_transforms=["cellpose"],
        train_data_path=img_patch_dir.as_posix(),
        valid_data_path=img_patch_dir.as_posix(),
        test_data_path=img_patch_dir.as_posix(),
        train_mask_path=mask_patch_dir.as_posix(),
        valid_mask_path=mask_patch_dir.as_posix(),
        test_mask_path=mask_patch_dir.as_posix(),
        type_classes=type_classes,
        sem_classes=sem_classes,
        return_inst=False,
        return_weight=False,
        batch_size=1,
        num_workers=1,
    )

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
