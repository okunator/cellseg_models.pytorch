import pytest

from cellseg_models_pytorch.utils.seg_benchmark import SegBenchmarker

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
@pytest.mark.parametrize("inputf", ["h5", "folder"])
def test_sem_seg_bm(mask_patch_dir, hdf5db, how, inputf):
    if inputf == "folder":
        inpath = mask_patch_dir
    else:
        inpath = hdf5db

    bm = SegBenchmarker(
        true_path=inpath,
        pred_path=inpath,
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
