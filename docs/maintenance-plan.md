# Repository maintenance plan

Modernize the library through small, independently reviewable changes while
preserving public APIs, pretrained checkpoints and numerical results. Code and
configuration define current behavior; this plan does not enable future features.

## Current baseline

Reconciled on 2026-10-12 against main at `b2b158e`, version `0.1.30`.
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
  Replacement examples and user guides are deferred to the docs-pages phase.
- uv 0.11.4 and `uv.lock` replace Poetry CLI (110); poetry-core 2.2.1 remains the
  build backend. Migration retained all 128 active dependency versions and verified
  artifact contents, source tests and sampled checkpoint predictions.
- Ruff is the single formatter/linter. Strict mypy checks the release guard and
  shared convolution helper (97/111), with no numerical behavior changes.
  Broader library typing and `py.typed` remain future work.

All 13 hosted checks passed for the latest typing patch:
[PR 111 validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38296855470).
Completed investigations and advisory details live in their PRs and Git history,
not a second review diary in this repository.

## Next work, in order

1. Finish the cleanup audit below, retaining maintained regressions and actionable
   guidance. Keep tooling, runtime, typing and structural changes independently
   reviewable.
2. Validate a release candidate against the affected checkpoint, optional backend,
   numerical and training paths. Refresh dependency advisory triage. Resolve any
   concrete failures before advancing; sampled CPU evidence does not establish
   representative accuracy, full-slide execution or GPU/AMP compatibility.
3. Agree on the release version and compatibility notes, then update manifest,
   module version and lockfile together using [the release guide](releasing.md).
   Publication and tags require maintainer authorization. Verify trusted publishing
   on that release and install the published package before closing issue 68.
4. Build documentation pages and verified CPU inference/training/fine-tuning
   examples after maintenance. Revisit the deferred structure proposals separately.

## Remaining validation and compatibility work

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
- [ ] Expand library annotations/stub coverage incrementally without blanket
  suppressions, casts or altered runtime representations. Package `py.typed` only
  after the supported surface and downstream distribution checks are ready.
- [ ] Review issue 78 without silently changing preprocessing or licensing.
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

No open GitHub dependency alerts were reported on 2026-10-12 after PR 105.
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
  imports, serialization and CPU training checks pass. Hosted clean-install gates
  remain required before merge.
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

User documentation pages cover quick start, training/fine-tuning (76),
model/checkpoint/backbone guidance (74), in-memory inference (69) and optional
packages. Introduce working examples and the smallest useful validation then.
Keep the README quick start. 3D support and new architectures are separate features.

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and independent feature installs. It remains deferred structural work, not a
prerequisite bundled into dependency upgrades. The GPU post-processing proposal
is preserved on `docs/gpu-postprocessing-proposal` for work after maintenance.
Retire or consolidate deferred drafts when adopted or cancelled.
