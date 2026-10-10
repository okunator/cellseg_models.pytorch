# Draft GPU postprocessing proposal

Status: deferred until the maintenance baseline is complete. Prepared 2026-10-10.
This is a proposal for later planning, not an implementation task or an approved
API change.

Keep dense predictions on the accelerator through instance reconstruction, then
transfer final labels or compact object information only when the caller needs
CPU results. The goal is lower end-to-end latency and higher sustained throughput
without changing segmentation semantics, checkpoint behavior, or memory bounds.
The expected benefit is a hypothesis to measure, not a claim that every GPU port
will be faster.

## Current execution and transfer boundaries

`BaseModelInst.post_process` defaults to `use_async_postproc=True` and
`start_method="threading"`. Its parallel paths use MPIRE workers; asynchronous
submission describes host task scheduling, not device execution.

`PostProcessor._prepare_inst_maps` converts dense maps to NumPy through
`_to_ndarray`, which moves CUDA tensors to CPU. Model-specific functions then
receive arrays. Some code moves selected arrays back to a GPU: CellPose's flow
integrator has an existing Torch CUDA path but converts NumPy inputs to tensors
and returns NumPy again. This is an especially useful transfer boundary to profile.
Current InstanSeg postprocessing creates CPU tensors from its NumPy inputs.

Postprocessing also includes type assignment, filtering, optional smoothing,
vectorization, and serialization. Accelerating reconstruction alone may leave
these stages as bottlenecks. Inventory the complete path for every model,
including nuclei, cytoplasm, tissue outputs, and tile/slide aggregation.

## Proposed PostProcessor changes

Keep the public orchestration role and the existing CPU behavior. Introduce an
explicit tensor/device contract at the model-specific postprocessing boundary,
so supported implementations receive tensors on the model's device. Preserve
dtype, axes, logits/probability semantics, class mappings, and instance/background
conventions. Define which operations return labels and which save outputs; the
current list/NumPy return surface needs a compatibility adapter at the actual
output boundary rather than an early conversion.

Separate numerical reconstruction from scheduling and CPU export. The CPU path
can continue using MPIRE. The GPU path should initially execute in one process
on the caller's device, reuse the caller's inference context, and process bounded
batches. Keep file writing and geometry conversion on CPU where appropriate.
Avoid a universal GPU worker framework or a new result hierarchy before a first
model port demonstrates what is needed.

Unsupported algorithms or devices must have an explicit documented CPU path.
Define fallback behavior and output placement before implementation; do not hide
host transfers inside a function advertised as device-resident. Preserve existing
callers while offering an opt-in tensor output if its use case justifies an API
addition. A backend/device selector is a design question, not a settled signature.

## MPIRE and GPU concurrency

MPIRE provides host processes or threads. A worker can call a GPU library, but
its task queue does not itself provide batching, stream dependencies, or GPU
memory management. Replacing NumPy with Torch inside an unchanged worker pool
would not address those concerns. [MPIRE start methods](https://sybrenjansen.github.io/mpire/usage/workerpool/start_method.html).

CUDA operations are normally enqueued asynchronously. A stream orders device
work; independent streams may overlap when resources permit. Host threads do
not automatically create useful GPU concurrency. Side streams require explicit
dependencies and tensor lifetime management. Begin with one stream; measure
before adding overlap between prediction, reconstruction, and transfers.
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).

Do not inherit an initialized CUDA runtime through `fork`. CUDA subprocess use
requires `spawn` or `forkserver`, with additional ownership/lifetime constraints
for shared tensors. If multiple GPUs later require process orchestration,
evaluate one process per GPU separately from the single-GPU implementation.
[PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html).

## Model feasibility inventory

These are candidates from source inspection, not measured rankings.

| Model family | Likely tensor operations | Harder reconstruction or parity work |
| --- | --- | --- |
| CellPose and Omnipose | Flow sampling/integration, normalization, thresholds, reductions | Convergence, grouping endpoints into instances, hole filling, small-object cleanup, border behavior; preserve existing integration precision. |
| HoVerNet, DCAN, DRFNS | Gradients, filtering, thresholds, morphology | Connected components and watershed, including connectivity and tie-breaking; a similar-looking replacement is not sufficient. |
| StarDist and CPPNet outputs | Candidate scoring, ray geometry, bounding boxes | Polygon overlap/NMS and rasterization; current StarDist uses CPU geometry and Numba/KDTree operations. Consider a measured hybrid path using compact candidates rather than dense maps. |
| InstanSeg | Seed detection, learned pixel classification, crop scoring, sparse overlap/merging | Correct the contributed training/classifier and seed semantics first. Upstream already offers a GPU/TorchScript reference; compare against it rather than accelerate the current incomplete implementation. |
| Shared output processing | Label counts, class voting, filtering and selected reductions | Preserve class ties and object identities; vectorization and file output may sensibly remain CPU operations. |

Most arithmetic and filtering should be feasible on GPU. Connected components,
watershed, sparse graphs, polygon geometry, and dynamic object counts need
individual algorithm and dependency decisions. Prefer existing PyTorch operations
first. Evaluate an external GPU library or custom kernel only for a measured gap,
with wheel availability, license, supported devices, and maintenance cost recorded.
CUDA success does not establish MPS or other accelerator support.

## Correctness and memory requirements

Use immutable checkpoints and preprocessing from the completed maintenance
baseline. Fix known correctness defects before treating an implementation as the
reference. For InstanSeg, resolve the supplied loss's missing classifier/sigma
training path and establish upstream prediction equivalence separately.

Compare dense outputs where applicable and reconstructed masks, counts, types,
coordinates, and pixel calibration. Compare instance partitions independently of
arbitrary label numbering, while preserving any externally meaningful identifiers.
Set tolerances before measuring. Test empty images, a single instance, touching
objects, high object density, edges, odd sizes, ties, batches, and invalid inputs.

Bound memory for seed crops, object-pair comparisons, and slide processing. Avoid
unbounded object-by-image tensors and all-pairs matrices. Profile synchronization
from scalar reads, dynamic shape operations, and host conversions. Precision
choices must preserve discrete segmentation decisions; model AMP settings should
not silently determine reconstruction precision. Test supported precision/device
combinations explicitly.

## Proposed delivery sequence

1. Complete maintenance and establish trustworthy CPU/reference inference tests.
   Revisit this proposal with hardware and compute budget agreed.
2. Profile representative sparse/dense tiles and slides. Measure prediction,
   device transfers, reconstruction, class assignment, export, and total latency.
   Record transfer bytes, batch/input sizes, object density, CPU/GPU peak memory,
   warmup, repetitions, package versions, and hardware. Use correct device timing.
3. Choose one vertical pilot from the measured bottlenecks. CellPose's existing
   GPU integration is a candidate; corrected upstream-equivalent InstanSeg is
   another. Define acceptance criteria before implementation.
4. Port that model's complete numerical path and make the output boundary
   explicit. Establish parity and memory bounds in the initial single-stream
   implementation, retaining CPU coverage and the current default behavior.
5. Add further models separately, prioritizing measured end-to-end benefit.
   Introduce concurrency, compilation, or custom kernels only where profiling
   justifies their additional contracts and complexity.
6. Evaluate defaults only after correctness, small-input crossover, sustained
   slide throughput, and peak memory are verified on supported hardware. Keep
   explicit CPU operation and a reversible rollout.

## Decisions to revisit

- Target CUDA first, or require MPS/other devices in the first release?
- Which public callers need device tensors, and which require NumPy or files?
- Which representative workloads determine performance acceptance criteria?
- Which connectivity, tie-breaking, and label identity rules must match exactly?
- Is corrected InstanSeg integrated from upstream or maintained as a faithful port?
- Which operations justify a hybrid CPU stage, external dependency, or custom kernel?

No speedup target, dependency choice, concurrency API, or default switch is
committed by this draft. GPU migration remains separate from dependency upgrades,
typing rollout, uv migration, and current maintenance PRs.
