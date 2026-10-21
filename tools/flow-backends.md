# Compare original and tensor reconstruction

The original CellPose/Omnipose NumPy integrator remains in
`cellseg_models_pytorch/postproc/functional/cellpose/integrator.py`.
Its functions and their original behavior are preserved. This includes the
historical single-seed shape limitation and silent interpolation failure path.
The independent tensor implementation lives in `postproc/torch/flow.py` and is
available as `cellseg_models_pytorch.postproc.follow_flows_torch`.
Existing NumPy reconstruction continues to select the original implementation.

Run the bounded comparison from the repository root with the locked development
environment. The data directory must contain the named dataset folders below:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  uv run --locked python tools/benchmark_flow_backends.py \
  --data-root /path/to/data --output /tmp/flow-backends-new-run
```

The output directory must be new. Numba JIT must be enabled. The tool uses CPU
tensors, two Torch/Numba threads, one Torch interop thread, two warmups and seven
measured repetitions per backend and phase. Method order alternates each
repetition. Loading, prediction, target generation, map preparation, input
conversion, metrics and file I/O stay outside timed intervals. Mac MPS and
Apple's Neural Engine are not used; the integrators retain float64 sampling.

The fixed inputs are six evenly spaced source rows from each pinned dataset:

| Dataset | Local folder and file | Source partition |
| --- | --- | --- |
| [PanNuke](https://huggingface.co/datasets/csmp-hub/pannuke) | `pannuke-parquet/fold_3.parquet` | Original fold 3 |
| [PanopTILs Refined](https://huggingface.co/datasets/csmp-hub/panoptils_refined) | `panoptils-refined-splits/test.parquet` | Patient-grouped test |
| [CytoNuke](https://huggingface.co/datasets/csmp-hub/cytonuke) | `cytonuke-parquet/test.parquet` | Added patient-grouped test |

Dataset revisions and full Parquet hashes are pinned in the tool and recorded
in the report. Verify downloaded files against those hashes before using an
existing cache. Dataset cards retain source attribution and license terms:
PanNuke is CC BY-NC-SA 4.0; PanopTILs is CC0; CytoNuke is CC BY 4.0 with upstream
CPTAC images under CC BY 3.0. This comparison does not redistribute their pixels.

Take centered 256-pixel and, where available, 512-pixel crops at native resolution,
without resizing. IDs and cropped object fragments are retained. This yields
24 crops with the default six-row setting: six PanNuke, twelve PanopTILs and six
CytoNuke. Each crop has three cases:

- CellPose reconstruction from target masks converted with `gen_flow_maps`.
- CellPose reconstruction from a pinned HGSC model's predictions.
- Omnipose reconstruction from target masks converted with `gen_omni_flow_maps`
  and the existing divergence rescaling.

The public HGSC checkpoint is pinned by revision and SHA256 and loaded through
safetensors. It runs float32 inference on CPU, batch one, without AMP, on globally
min-max-normalized RGB crops. Its seven classes differ from the dataset taxonomies,
so the tool compares instance quality rather than cross-dataset cell-type scores.
There is no compatible trained Omnipose checkpoint in this comparison; Omnipose
prediction quality is untested. These are cross-dataset reconstruction checks,
not evidence that the HGSC model generalizes to these datasets or that their
splits are independent of its original training data.

Both implementations receive identical prepared flow fields, masks and class
maps. They use 200 iterations, float64 integration and float32 endpoints.
Target round trips keep small objects (`min_size=0`); predictions use the existing
`min_size=30`. Both use the same CPU grouping, hole filling and class assignment.
The two timing phases are integration alone and complete reconstruction from
these prepared flows, including grouping, cleanup and voting. The latter is not
full image inference or the high-level worker/file-export API.

Every case retains its input field and both outputs in a compressed NPZ, with a
hash recorded in `report.json`. `cases.jsonl` is flushed as work completes.
The report includes raw timing samples, medians, source identity, CPU/thread and
package details, annotation density and object counts. Exact endpoint values,
seed indices, background/object partitions (allowing positive ID permutations)
and class masks are checked. Quality is per-crop PQ/DQ/SQ at strict IoU > 0.5.
Empty masks retain the metric's defined zero score. Partial annotations and cut
objects limit quality interpretation. Never relax parity to make a run pass;
differences are saved and the command fails after writing its report.
Process peak RSS includes imports, model loading/inference, target generation
and both backends; it is not a per-backend memory comparison.

Keep raw reports and prepared arrays while the comparison is under review.
Retire temporary smoke outputs and rendered summaries once a retained report
and maintained tests own their evidence. Retain this tool while its pinned
comparison is useful; replace it when a broader benchmark supersedes the scope.
No new routine dataset-download or benchmark CI job is added.
