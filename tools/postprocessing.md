# Post-processing migration baseline

Replay preserved CellPose and StarDist dense predictions without loading a model
or downloading weights. This is the first measurement tool for the
[GPU migration proposal](https://github.com/okunator/cellseg_models.pytorch/blob/docs/gpu-postprocessing-proposal/docs/gpu-postprocessing-draft.md).
It measures existing behavior; it does not provide a new GPU backend.

From the repository root, with the locked development environment installed:

```sh
HF_HUB_OFFLINE=1 NUMBA_NUM_THREADS=2 uv run --locked python tools/profile_postprocessing.py --family cellpose --device cpu --output /tmp/cellpose-cpu.json
HF_HUB_OFFLINE=1 NUMBA_NUM_THREADS=2 uv run --locked python tools/profile_postprocessing.py --family stardist --device cpu --output /tmp/stardist-cpu.json
```

Use a new report filename for each run. Reports are created exclusively, so
existing measurements cannot be overwritten. Numba JIT must be enabled for timing;
warmup includes compilation. Defaults are two warmup runs, five measured runs and
two Torch threads. Thread counts and package versions are recorded in JSON.
The report also records UTC time, source commit and whether the checkout has
uncommitted changes. Record the CPU model and power mode alongside reports when
the platform's processor name is incomplete.

On an authorized CUDA machine, first check active jobs and available memory with
`nvidia-smi`, and agree a bounded compute budget. Run the same command with
`--device cuda:0` and a different output path. The selected CUDA device controls
both the source prediction tensors and CellPose's existing flow integrator.
CUDA requests fail if unavailable; CPU runs keep integration on CPU even on a
GPU machine. MPS and other devices are not covered.

The input is one 256 × 256 tile with float32 dense reconstruction values and
integer class labels derived from the pinned class logits. Channel order and
head interpretation follow the
[checkpoint fixture contract](../tests/fixtures/checkpoint_baseline/README.md).
CellPose flow integration retains the existing float64 arithmetic. Every warmup
and measured run verifies exact object partitions, background, classes and mask
dtypes against the checksum-verified reference. Positive instance IDs may permute;
splits, merges or moved boundaries fail the run.

The report separates these wall-clock measurements, synchronizing the selected
CUDA device before and after each interval:

- `prepare`: the real `PostProcessor._prepare_inst_maps` call, including device
  to CPU copies, dtype conversion and array/list construction.
- `reconstruct`: the configured reconstruction function and int32 output conversion.
  On CUDA, CellPose currently copies arrays back to the GPU and endpoints back to
  CPU here; StarDist reconstruction stays on CPU.
- `classify`: the existing per-instance majority vote and int32 conversion.
- `total`: a separate call to the real serial post-processing API, including its
  preparation, reconstruction, classification and garbage collection.

Stage timings are diagnostic replays, not a partition of `total`; they exclude
orchestration and garbage collection, and synchronization changes overlap.
Raw samples and medians are retained. Input upload and fixture loading are outside
the timings, matching predictions already resident on the model's device.
`prepare_device_to_host_bytes` counts source tensor bytes copied by preparation,
including the duplicated class-map copy when there is no binary head. It excludes
the integrator's internal transfers. `prepared_array_bytes` counts resulting NumPy
arrays. CUDA peaks include resident input tensors and reconstruction, after warmup;
reserved memory may retain warmup allocations. Process peak RSS on Unix is the
whole process lifetime high-water mark, including imports, loading and warmup,
not reconstruction-only memory. Memory is not sampled per stage.

This fixture is a correctness and measurement smoke test, not representative
performance evidence. It excludes model prediction, MPIRE scheduling, cytoplasm,
tissue, geometry/file export and slide aggregation. Before choosing a vertical
GPU pilot, capture representative sparse/dense, empty, border and odd-sized tiles,
multiple batch sizes and slide workloads. Measure the omitted stages, transfer
traffic and bounded peak memory on the target hardware; fix parity tolerances
before implementing a port. Keep CPU behavior and defaults unchanged. CellPose
flow integration is a candidate, not a measured winner.

This repository-only tool and its tests are not library API or distribution
contents. Retire or fold it into a broader benchmark when representative model
and slide measurements replace this initial fixture replay. Keep raw reports with
the corresponding PR or experiment record; do not commit machine-specific timings.
