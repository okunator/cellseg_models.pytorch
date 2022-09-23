from cellseg_models_pytorch.inference import ResizeInferer
from cellseg_models_pytorch.models import cellpose_plus
from cellseg_models_pytorch.utils.latency_benchmark import LatencyBenchmarker


def test_latency_benchmark(img_dir):
    model = cellpose_plus(sem_classes=3, type_classes=3, long_skip="unet")

    inferer = ResizeInferer(
        model,
        img_dir,
        out_activations={"sem": "softmax", "type": "softmax", "cellpose": "tanh"},
        out_boundary_weights={"sem": False, "type": False, "cellpose": True},
        resize=(256, 256),
        padding=80,
        instance_postproc="hovernet",
        batch_size=1,
        save_intermediate=True,
        device="cpu",
        parallel=False,
    )
    inferer.infer()

    bm = LatencyBenchmarker(inferer)

    bm.postproc_latency("inst", reps_per_img=1)
    bm.postproc_latency("type", reps_per_img=1)
    bm.postproc_latency("sem", reps_per_img=1)
    bm.inference_latency(reps=1, warmup_reps=0)
    bm.inference_postproc_latency(reps=1)
    # bm.model_latency(input_size=(64, 64), reps=1, warmup_reps=0, device="cpu")
    # bm.model_throughput(input_size=(64, 64), reps=1, warmup_reps=0, device="cpu")
