import pytest

from cellseg_models_pytorch.inference import ResizeInferer, SlidingWindowInferer
from cellseg_models_pytorch.models import cellpose_plus


@pytest.mark.parametrize("batch_size", [1, 2])
def test_slidingwin_inference(img_dir, batch_size):
    model = cellpose_plus(sem_classes=3, type_classes=3, long_skip="unet")

    inferer = SlidingWindowInferer(
        model,
        img_dir,
        out_activations={"sem": "softmax", "type": "softmax", "cellpose": "tanh"},
        out_boundary_weights={"sem": False, "type": False, "cellpose": True},
        patch_size=(256, 256),
        stride=256,
        padding=80,
        instance_postproc="hovernet",
        batch_size=batch_size,
        save_intermediate=False,
        device="cpu",
        use_blur=True,
        use_closing=True,
    )

    inferer.infer()

    samples = list(inferer.out_masks.keys())
    assert inferer.out_masks[samples[0]]["inst"].shape == (512, 512)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_resize_inference(img_dir, batch_size):
    model = cellpose_plus(sem_classes=3, type_classes=3, long_skip="unet")

    inferer = ResizeInferer(
        model,
        img_dir,
        out_activations={"sem": "softmax", "type": "softmax", "cellpose": "tanh"},
        out_boundary_weights={"sem": False, "type": False, "cellpose": True},
        resize=(256, 256),
        padding=80,
        instance_postproc="hovernet",
        batch_size=batch_size,
        save_intermediate=False,
        device="cpu",
        use_blur=True,
        use_closing=True,
    )

    inferer.infer()

    samples = list(inferer.out_masks.keys())
    assert inferer.out_masks[samples[0]]["inst"].shape == (512, 512)
