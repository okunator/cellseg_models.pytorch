# Repository maintenance plan

Modernize the library through small, independently reviewable changes while
preserving public APIs, pretrained checkpoints and numerical results. Code and
configuration define current behavior; this plan records remaining work.

## Current baseline

Reconciled on 2026-10-17 against main at `ffbb4df` after PR 113, version `0.1.30`.
The maintainer chose validation only: no version bump, tag or publication is
scheduled. Python 3.10 remains the minimum; source and clean wheel/source installs
are tested on 3.10–3.12. Newer versions remain unvalidated, and torchvision's
Python 3.14.1 exclusion still applies.

All 13 checks pass on [merged main](https://github.com/okunator/cellseg_models.pytorch/actions/runs/39023542668),
including source tests, clean wheel/source installs, ONNX and the docs build.

Completed:

- ONNX exports, direct/optional import repairs, dependency modernization,
  normalization and metric fixes, and HDF5 alignment/count repairs are merged.
- uv 0.11.4 and `uv.lock` replace Poetry CLI; poetry-core remains the build backend.
  Ruff is the formatter/linter. Strict mypy checks the release guard, convolution
  helper and WSI instance merger; broader typing remains incremental.
- [PR 113](https://github.com/okunator/cellseg_models.pytorch/pull/113) adds the
  Sphinx/PyData site and four maintained notebooks: PanopTILs preparation, training
  with optional encoder freezing, image inference and WSI inference. Reviewed
  outputs come from real data and DGX execution. Contributor and release guidance
  stay outside the user site.
- PR 113 also fixes DiceLoss and augmentation alignment, reconciles whole source
  nucleus polygons while preserving unmatched detections, and adds opt-in vector
  filtering, calibrated reader metadata and cuCIM GPU tile loading. Checkpoints
  remain loadable; the corrected Dice objective changes training behavior.
- Real OpenSlide/cuCIM runs cover 804 tiles and twelve reviewed seam fields.
  The 3,216-tile CPU merge/export stress check uses about 6.3 GiB; model/CUDA
  allocations are excluded and memory still grows with source polygons. Detailed
  timings, quality comparisons and their limits remain in PR 113.

## Remaining documentation

- [ ] Add concise model/checkpoint/backbone guidance, including class mappings and
  preprocessing, then a focused public API reference.
- [ ] Document the HDF5 training alternative without duplicating the notebook loop.
- [ ] Document supported ONNX and device/precision options where users need them.
  Keep real GPU/checkpoint runs explicit and preserve reviewed notebook outputs.
- [ ] Show sliding-window inference on an image already in memory (issue 69).

The four notebooks and their tests are the maintained workflow guidance. Extend
examples only when a different model or input contract needs one. The training
example supplies the workflow requested in issue 76. Issue 74's backbone question
is closed; the broader guide above remains documentation work.

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
- [ ] Extend optional-backend coverage only when needed by a supported workflow.
  OpenSlide/cuCIM full-slide inference and the documented CUDA training path are
  validated in PR 113; BioIO calibration has mocked coverage only. HDF5 repairs
  have real PyTables/Albumentations checks. Broader GPU/AMP support remains open.
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

## Cleanup and release

- [x] Retire PR 113's temporary comparison galleries, superseded notebook copies,
  profiling scripts and diagnostic results locally and on the DGX. Keep raw data,
  trained weights/resume state, maintained notebooks and regression fixtures.
  Remove the completed docs checklist and investigation diary from this plan;
  PRs and Git history preserve the evidence.
- [x] Rebuild and inspect wheel/source contents after PR 113. Tests, fixtures,
  notebooks, temporary docs, logs and diagnostic data remain outside distributions.
- [x] Review CI cost: source, clean installs, minimum/current ONNX and release checks
  retain distinct contracts. Docs rendering and merger typing reuse the Python
  3.11 job; no recurring dataset/checkpoint download or GPU job is required.
- [ ] When the maintainer requests a release candidate, refresh dependency alerts,
  check version/tag agreement and migration notes, and validate exact wheel/source
  artifacts outside the checkout. Preserve prior releases and checkpoints.
- [ ] Verify configured PyPI trusted publishing on an authorized release, publish
  the verified artifacts and check the resulting installation, including issue 68.
  Issue 73 is closed and its metric fix is on main; released availability remains
  pending.

After each batch, remove temporary artifacts once maintained tests or guidance
cover their purpose. Retire a CI check only when its contract is obsolete or
covered elsewhere. Keep release instructions in [the release guide](releasing.md).

## Deferred work

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and independent feature installs. The GPU post-processing proposal is preserved
on `docs/gpu-postprocessing-proposal`. Both remain separate structural work;
retire or consolidate their drafts when adopted or cancelled. 3D support and
new architectures are separate features.

Probability export and changes to model supervision remain deferred. PanopTILs
class 0 includes untyped nuclei, so its current type head cannot supply calibrated
foreground as `1 - P(background)`. Preserve checkpoint and standalone reconstruction
contracts without requiring flow outputs. Revisit the binary-only forward contract
and `save_intermediate` aliasing only if those paths enter scope; optional geometry
filtering remains disabled by default.
