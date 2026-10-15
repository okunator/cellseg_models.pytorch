# Repository maintenance plan

Modernize the library through small, independently reviewable changes while
preserving public APIs, pretrained checkpoints and numerical results. Code and
configuration define current behavior; this plan does not enable future features.

## Current baseline

Reconciled on 2026-10-13 against main at `4d5af51`, version `0.1.30`.
The maintainer chose validation only: keep the version unchanged; no tags or publication.
No maintenance release has been published. Python 3.10 remains the minimum;
source and clean wheel/source installs are tested on 3.10, 3.11 and 3.12.
Python 3.14.1 is excluded by the torchvision requirement; newer Python versions
are not yet validated.

Completed work:

- Contributor ONNX exports and their batching, dtype and state-restoration fixes
  are merged (80/81/87). Current CI exercises the minimum and current supported
  model stacks with explicit optional packages and rejects skipped comparisons.
- CI validates source tests, compiled numerical helpers, clean distributions and
  release metadata. Publication uses the exact verified artifacts and PyPI OIDC.
- Direct runtime imports and optional dataset/WSI import repairs are merged
  (89/90/91). CPU tiled WSI execution has maintained regressions (98).
- HTTP/cache, Pillow, geometry, model and numerical/image dependency batches are
  merged (93/94/96/102/103). Unnecessary transitive floors were removed (105).
- Shared normalization and DRFNS saturation, low-IoU metric assignment/FP/FN/sparse
  IDs, and HDF5 sample counts/target alignment are fixed (104/106/109).
- Legacy notebooks are retired; the README quick start is preserved (107/108).
  Replacement examples and user guides are the next phase below.
- uv 0.11.4 and `uv.lock` replace Poetry CLI (110); poetry-core 2.2.1 remains the
  build backend. Migration retained all 128 active dependency versions and verified
  artifact contents, source tests and sampled checkpoint predictions.
- Ruff is the single formatter/linter. Strict mypy checks the release guard and
  shared convolution helper (97/111), with no numerical behavior changes.
  Broader library typing and `py.typed` remain future work.

Cleanup is merged (112). All 13 checks passed on merged main:
[main validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38381936618).
The final CPU pass checked both pinned checkpoints from source and an installed
wheel, 15 optional HDF5 cases, 37 compiled numerical cases and four sampled decoder
forward/backward cases. No open dependency alerts were reported on 2026-10-13.
This is validation of current main, not a versioned release candidate.
Completed investigations and advisory details live in their PRs and Git history,
not a second review diary in this repository.

## Next work: documentation

Dependency maintenance is complete. Build and validate the examples against
current main; distinguish its unreleased requirements from PyPI.
Fix concrete failures in the paths being documented, without reopening deferred
refactors. Keep the README quick start and link it to the fuller guide.

### Small docs-pages plan

| Page | Content |
| --- | --- |
| Start here | Installation, tested Python/device support, CPU quick start, current main versus published release |
| Inference | RGB HWC images versus BCHW batches, checkpoint loading, exact normalization, prediction and instance/type masks; tiled inference and output saving |
| Models and checkpoints | Implemented architectures versus original papers, encoder choices, available weights, class IDs and checkpoint compatibility; do not invent semantic class names |
| Data and training | Folder/HDF5 layout, instance/type labels, aligned augmentation, model-specific targets, losses and a plain PyTorch training/validation loop |
| Optional workflows | Explicit requirements and tested limits for WSI backends, ONNX export and GPU/AMP; avoid unsupported performance/accuracy claims |
| API reference | Focused public entry points with Google-style docstrings and tensor contracts |

Keep contributor and release guidance in the repository, excluded from the
user-facing Sphinx site.

### Data preparation and runnable examples

Use maintained Jupyter notebooks under `docs/tutorials/`, rendered directly as
pages with Sphinx, MyST-NB and the PyData theme; do not duplicate their code in scripts. Begin with CellPose; add another model only
when its distinct target/output contract needs an example.

1. **Data preparation:** pinned full PanopTILs archive, patient-level split before
   aligned native-scale patches, instance and cell-type labels only, and a visible
   label/class-distribution check. Keep original held-out regions for evaluation.
2. **Inference:** one RGB image and a local or explicitly downloaded checkpoint,
   explicit GPU preference with CPU fallback and checkpoint preprocessing, then save instance
   and type masks. Explain the returned head keys, shapes, dtypes and background.
3. **Training:** include optional encoder freezing and use the Histolytics nuclei
   recipe through existing public APIs:
   pretrained image encoder, aligned augmentation, MSE flows, CE with spectral
   decoupling plus Dice cell-type loss, AdamW and a cosine schedule. Evaluate
   reconstructed instances and per-class quality, not just losses/dense labels.

All examples document input layout, optional packages, output paths and commands.
Use synthetic tiny data for offline checks; real data/checkpoints are explicit,
with attribution and patient/slide-independent splits. Do not present synthetic
loss reduction as segmentation accuracy or a useful trained model.

### Delivery and validation

- [x] Establish Sphinx/PyData/MyST-NB navigation, installation guidance and the
  inference notebook. A separate docs group preserves all prior locked versions.
  Navigation starts with installation and quick start, then groups notebooks in
  the user guide. Build instructions stay in contributor guidance.
- [x] Maintain offline training and optional-freezing regressions: finite losses
  and gradients, parameter updates, frozen encoder state and exact checkpoint reload
  are validated in `tests/test_notebook_tutorials.py`, outside the rendered pages.
- [x] Fix augmentation defects exposed by the real recipe: generate auxiliary
  targets after aligned image/mask transforms and compose all sampled image
  operations. Folder regressions, 24 optional HDF5 cases and the ordinary suite pass.
- [x] Prepare all 1,349 pinned PanopTILs regions: patient-separated splits,
  17,536 training and 4,048 validation patches at native scale, nucleus instances
  and cell types only. Dataset construction lives in the preparation notebook,
  with nucleus counts per cell type before cropping. Its offline tests execute
  those cells directly; raw data stays outside Git and distributions.
  Use one headless OpenCV provider for runtime and Albumentations. The existing
  installed-package gate rejects overlapping providers; runtime/docs sync round
  trips and both pinned CPU checkpoint predictions are verified.
- [x] Validate five full-data training epochs with the Histolytics nuclei recipe
  and corrected spatial DiceLoss. Twenty-one numerical CPU cases and the explicit
  CUDA loss check cover reduction, rare/absent classes, weighting, masking,
  smoothing, gradients and precision. On 16 held-out regions, instance PQ is 0.643;
  rare-class support/weakness remains visible. Model weights use safetensors beside
  separate resume state. The standalone fine-tuning notebook is retired in favor
  of a short optional-freezing section; tests cover frozen parameters and BatchNorm
  statistics. Inference uses the training checkpoint. Raw logs and resumable
  checkpoints stay outside Git. Preserve historical-objective checkpoints until
  the maintainer accepts the corrected examples.
- [ ] Document the HDF5 alternative without duplicating the training loop.
- [x] Execute the full WSI notebook on TCGA-AC-A6IW with the PanopTILs checkpoint:
  453 tissue tiles and 174,013 merged nuclei on DGX CUDA. Saved tissue-selection,
  overview and seam overlays accompany offline merge regressions; all exported
  geometries and slide-wide IDs pass the full-slide audit. The page keeps a
  tissue-edge false positive visible rather than implying whole-slide accuracy.
  Per-tile outputs and the public slide stay outside Git; retain them until the
  example is accepted, then retire redundant run artifacts.
- [ ] Add model/checkpoint and optional-workflow guidance, then the focused API
  reference. Keep the current normalization helper; issue 78's replacement/license
  proposal does not justify a silent preprocessing or dependency change.
- [x] Execute inference with the learned PanopTILs checkpoint and held-out labels
  at their original 1024-pixel scale. Keep the independent public-checkpoint quick
  start and the original 256-pixel numerical regression inputs/references unchanged.
- [x] Build with warnings treated as errors and render reviewed notebook outputs
  in CI. Execute all real-data/checkpoint notebooks explicitly outside offline CI.
  Model notebooks execute on the DGX Spark CUDA GPU; preparation runs on CPU. Reviewed cell
  outputs are retained without machine-specific paths. CPU checks stay in ordinary
  CI; CUDA notebook regressions use the existing `--cuda` category manually.
  Labelled figures show the input and masks; tests verify plotted labels, including
  foreground class 0, outside the pages.
  Reuse the Python 3.11 source job, with no new job or recurring weight downloads.
  Keep real-checkpoint, GPU and full-slide runs explicit, with skips/limits reported.

After adoption, remove this temporary docs checklist; the pages and examples become
the maintained guidance. Release validation below stays separate from docs delivery.
Retire the superseded four-region execution artifacts after the full-data runs are
verified; preserve actual checkpoint regressions and this workflow's resumable state.

## Remaining validation and compatibility work

- [ ] Correct IoULoss and TverskyLoss class-axis reductions with independent
  numerical regressions before recommending them; the current examples use neither.

- [ ] Verify published installation resolves issue 68. Python 3.12 source and
  clean-distribution checks already pass; a published-release check is still needed.
- [ ] Exercise representative images/model families, registered-name checkpoint
  loading and reconstructed masks beyond the two maintained checkpoint fixtures.
  Define tolerances before comparisons; never regenerate references to hide errors.
- [ ] Restore current image-inference coverage and select a small useful subset
  of the 5,184 gated decoder cases. Define a feasible separate slow/GPU schedule.
- [ ] Verify supported GPU/AMP, precision and driver paths with available authorized
  hardware. CPU checks and export sample construction do not validate these paths.
- [ ] Validate real slide backends/full-slide merging and affected training paths.
  HDF5 target/count repairs are checked with real PyTables/Albumentations files;
  broader optional-backend support is not established by skipped tests.
- [ ] Complete optional dependency declarations and independently tested import
  boundaries for slide, training and export features. Do not remove mandatory
  packages until every supported feature install preserves its API and errors.
- [ ] Check dependency wheels and execution before adding newer Python CI versions
  or changing the minimum. Update metadata, workflow, Ruff target, README and lock
  together in an explicit support-policy patch; retain Python 3.10 for now.
  Aim for installation without a user-supplied upper Python cap. Metadata already
  permits 3.13/3.14 except 3.14.1; Torch 2.14.1 and torchvision 0.29.1 provide
  wheels for both. Validate clean installs, compiled helpers and model execution
  on 3.13/3.14 next; retain the upstream 3.14.1 exclusion until torchvision removes it.
- [ ] Expand library annotations/stub coverage incrementally without blanket
  suppressions, casts or altered runtime representations. Package `py.typed` only
  after the supported surface and downstream distribution checks are ready.
- [ ] Address issue 78 when updating the inference guide; retain existing
  preprocessing and licensing unless an explicit, validated change is agreed.
- [ ] Retire the five superseded Dependabot PRs (23/48/50/52/57) and configure
  future grouped weekly dependency/Actions updates in a separate policy patch.
  Their proposed versions are older than the merged upgrades.

Run `--slow`, `--optional` and `--cuda` separately: each selects only its category.
After installing undeclared optional packages, use `uv run --no-sync` so a sync
cannot remove them. Check execution and skips; passing ordinary CI is not evidence
for a skipped integration.

## Baseline and dependency policy

[Maintained checkpoint fixtures](../tests/fixtures/checkpoint_baseline/README.md)
pin two public checkpoint revisions/digests, one image, preprocessing and preserved
PyTorch arrays. Their integrity check runs offline in ordinary CI; explicit native
comparisons remain manual for numerical upgrades and releases. Fixtures are outside
the runtime package and are excluded from distributions. Keep them while these
maintained checks consume them; broader datasets and accelerators remain open.

Native CPU comparisons check dense physical outputs and class probabilities with
explicit tolerances, exact class masks and object partitions/background, allowing
only a bijection of positive instance IDs. This does not change production outputs.
ONNX checks independently compare raw heads and dense class decisions. Native
float32/float64 calibration over six images justified the StarDist class-head
budget of `atol=2e-4, rtol=1e-4` (101); foreground/ray bounds remain at `1e-5`.
Both pinned HE.png models preserve instance/type masks. The calibration is resolved;
it is not a representative accuracy benchmark or GPU claim.

Prefer updating the parent dependency that owns a transitive requirement. Retain
a direct transitive security floor only for a reachable vulnerable path that the
parent minimum does not protect. Requests permits older urllib3/IDNA on downloader
traffic, and PyTorch/Hub permit older filelock on checkpoint caches, including the
soft-lock fallback. Keep these three floors until parent requirements protect those
paths. fsspec/setuptools/Soup Sieve/Pygments vulnerable paths were not used by this
library; their patched lock versions remain without unnecessary runtime floors.
Published installs use parent requirements, not the project lockfile.

No open GitHub dependency alerts were reported on 2026-10-13 after PR 105.
Refresh advisory ranges and applicability before the release candidate; this dated
snapshot is not a continuing security certification. Hub's transport migration,
libpysal's Python minimum and OpenCV 5 need separate compatibility checks.

## Cleanup and release checklist

Remove obsolete diagnostics and investigation notes after their facts reach a
maintained regression, release note or contributor guide. PRs and Git history
preserve past evidence. Review after each batch; do not add another inventory file.

- [x] Consolidate this plan and remove the completed historical review record.
  Retain the baseline policy, open issues, deferred drafts and release instructions.
- [x] Inspect fresh wheel/source contents and sizes. Exclude 54 package-local
  test/fixture entries through the existing build backend, retaining checkout tests.
  The wheel shrinks from 4.04 MB to 271 KB and the source archive from 3.91 MB to
  164 KB; retained wheel payload bytes are identical. Local installed wheel/source
  imports, serialization and CPU training checks pass. All six hosted clean-install
  gates also passed on merged main.
- [x] Review CI timings, downloads and artifact retention. Retire a job only when
  its contract is obsolete or covered elsewhere. Preserve distinct source,
  installed-package, minimum/current export and release checks. Avoid recurring
  checkpoint downloads or large investigative matrices.
- [x] Check documentation links after consolidation and keep unresolved work explicit.
- [ ] Build the agreed release candidate, check version/tag agreement and migration
  notes, and install wheel/source artifacts outside the checkout. Preserve prior
  releases/checkpoints and keep changes revertible.
- [ ] Verify PyPI trusted publisher authentication on an authorized release; CI
  cannot verify account configuration. The maintainer reports configuring the
  matching publisher and `pypi` environment. Do not introduce a stored-token fallback.
- [ ] Publish the exact validated artifacts and verify the resulting installation.
  Metric issue 73 is fixed on main; its short issue comment is drafted for maintainer
  review, and availability in a release remains pending.

Latest pre-cleanup CI takes about 3½ minutes overall: source jobs 2½–3 minutes,
clean installs 2–3¼ minutes, ONNX jobs under 2 minutes, build 19 seconds and
standalone developer checks 12 seconds. These jobs still exercise distinct contracts.
The remaining tools are consumed by build/install/release gates; checkpoint fixtures
are consumed by offline integrity and manual comparison checks.

## Deferred work

The active docs phase above covers issues 69/74/76 and the quick start.
3D support and new architectures remain separate feature work.

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and independent feature installs. It remains deferred structural work, not a
prerequisite bundled into dependency upgrades. The GPU post-processing proposal
is preserved on `docs/gpu-postprocessing-proposal` for work after maintenance.
Retire or consolidate deferred drafts when adopted or cancelled.
