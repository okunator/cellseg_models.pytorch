import pytest

from cellseg_models_pytorch.utils.seg_benchmark import BenchMarker

classes = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
}


@pytest.mark.parametrize("how", ["binary", "multi", None])
def test_sem_seg_bm(mask_patch_dir, how):
    bm = BenchMarker(
        pred_dir=mask_patch_dir,
        true_dir=mask_patch_dir,
        type_classes=classes,
        sem_classes=classes,
    )

    if how == "binary":
        res = bm.run_inst_benchmark(how, metrics=("dice2",))
    elif how == "multi":
        res = bm.run_inst_benchmark(how, metrics=("dice2",))
    else:
        res = bm.run_sem_benchmark(metrics=("iou",))

    assert isinstance(res, list)  # convenience
