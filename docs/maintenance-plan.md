# Repository maintenance plan

Refresh the project in small, independently reviewable changes. Establish reliable
CI and reproducible predictions before upgrading runtime dependencies, so existing
users retain working APIs, checkpoints, and segmentation results.

## Current baseline

Status reconciled on 2026-10-09 against `main` at
`574a10d6c25fb091dc9f4b47798dcf53120ee625`, version `0.1.30`.
Maintenance PR 84, Google-style docstring PR 85, contributor ONNX PRs 80 and 81,
and corrective PR 87 are merged. PR 86 is closed because PR 87 included its
Google-style exporter docstrings. The old `fix/onnx-maintenance` checkout is a
historical review workspace; its corrections are now on main.

All nine checks passed on the merged main commit in
[run 37966314883](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37966314883):
source suites on Python 3.10/3.11, package build, four clean distribution installs,
and CPU ONNX checks on PyTorch 2.7.1/2.14.1. Ordinary inference requirements and
`poetry.lock` remain unchanged. ONNX export requires PyTorch 2.7 or newer.

The original contributor PRs passed ordinary tests but skipped runtime comparisons
without optional ONNX packages. They were merged before the corrective patch;
PR 87 supplied the batching/state/dtype fixes and explicit runtime checks. This
sequence deviated from the original merge gate and is now corrected on main.
Pretrained StarDist dense parity and representative prediction coverage remain
open; passing synthetic-model checks does not resolve those gaps.

Of the ordinary suite's skipped cases, 5,184 belong to a deliberately gated
decoder matrix. `--slow`, `--optional`, and `--cuda` each select only their own
category; run them separately from ordinary tests.

## Next work, in order

1. Completed in PR 89: repair the clean-install WSI Matplotlib import guard,
   add an absent-Matplotlib regression, and import WSI in wheel/source smoke
   checks. Continue coverage of concrete inference/data APIs below.
2. Finish direct-runtime dependency declarations and optional import boundaries.
   Reproduce issue 68 with the locked environment and clean range-based installs.
   Keep optional slide/training backends separate from mandatory runtime packages.
3. Refresh advisory triage before each small I/O or downloader security batch.
   Record affected paths and resolved advisory IDs; update manifest/lock together
   where needed. These focused fixes can precede the broader numerical baseline,
   with image-decoding/checkpoint checks appropriate to each changed dependency.
4. Complete the reproducible prediction baseline: immutable checkpoint/image
   identities, encoder/class/output settings, preprocessing, dense outputs,
   reconstructed masks and counts. Add representative images/models and investigate
   the existing StarDist ONNX tolerance failure without relaxing it for a green CI.
   A new scientific tolerance requires an independently justified validation change.
5. Modernize development tooling and introduce one explicit passing typing scope.
   These can proceed independently of the numerical baseline; keep them separate
   from runtime upgrades and the uv migration.
6. Upgrade model, numerical/image, and geospatial dependencies in separate batches
   once their baseline checks exist. Add newer Python CI coverage after dependency
   wheel/compatibility checks; change the Python floor only in an explicit support
   policy patch. Python 3.10/3.11 remain the tested source matrix today.
7. Migrate Poetry to uv after dependency audit and baseline validation, then
   complete release-candidate checks. Update user documentation and address
   installation/correctness issues alongside each relevant batch rather than
   waiting for uv. Keep trusted publisher authentication pending until an
   authorized publication verifies it.

Do not wait for merges to perform independent read-only audits. Keep each code,
tooling, or dependency batch independently reviewable; do not combine all phases
into one PR.

## First patch for CI and publication

- [x] Verify the locked environment on the existing Python 3.10 and 3.11 matrix.
- [x] Update Actions and pin Poetry. Always synchronize dependencies and install
  the current package; cache downloads rather than installed environments.
- [x] Preserve test and coverage reports, check that coverage was generated, and
  make coverage upload failures independent of test success. Fork PRs need no secrets.
- [x] Run CI for documentation and example changes as well as library changes.
- [x] Build the wheel and source distribution once, check their metadata, and
  install each into a clean environment outside the checkout. Exercise public
  imports and a CPU forward, backward, and optimizer step.
- [x] Gate publication on both the source suite and distribution checks. Require
  agreement between the tag, manifest, and module versions. Publish the exact
  validated artifacts; manual dispatch validates without publishing.

Keep runtime requirements, lockfile, public APIs, and Python support unchanged
in this first patch. Completion requires a passing hosted run, including the
checks that cannot be reproduced locally.

## Dependency audit and prediction baseline

- [ ] Triage the repository's dependency security advisories and prioritize fixes
  affecting reachable code paths. Include resolved advisories in each upgrade's
  validation record.
- [ ] Inventory direct imports, including Hugging Face Hub, safetensors, Pillow,
  and geospatial packages. Declare required packages directly or provide an
  explicit, documented extra with tested import boundaries.
- [ ] Reproduce installation issue 68 on supported platforms and Python versions.
  Test both the locked development environment and published dependency ranges.
- [ ] Record immutable checkpoint revisions or checksums, encoder configuration,
  class mapping, normalization, image identity, and model output settings.
- [ ] Capture representative dense predictions, reconstructed instance masks,
  and instance counts. Define tolerances before comparing dependency changes.
- [ ] Exercise real public checkpoint loading as well as local state dictionaries.
- [ ] Restore current image inference tests and a small representative subset of
  the large decoder matrix. Define a feasible separate slow and GPU test schedule.

Completion requires a clean installation and a reproducible inference baseline.

## ONNX contribution review

- [x] Review PRs 80 and 81 individually, preserving contributor attribution.
  See [the review record](maintenance-review.md) for confirmed blockers and prepared fixes.
- [x] Install ONNX, ONNX Script, and ONNX Runtime in an explicit validation job.
  Require that export and runtime comparison tests execute rather than skip.
- [x] Test batches of size one and greater than one, output names, fixed spatial
  dimensions, missing dependencies, export failure, model dtype, and restoration
  of model training state, including mixed child-module states.
- [ ] Repeat pretrained-image comparisons using the recorded checkpoint and image.
  Document that ONNX covers dense prediction; reconstruction remains in Python.
- [x] Check the declared minimum export PyTorch version and a current supported
  version. Keep export dependencies optional for ordinary PyTorch inference.
- [x] Merge the separate contributions and validate their combined corrected
  behavior through PR 87 and merged-main CI. The original merge sequence deviated
  from the planned ONNX gate, as recorded above. Shared-helper cleanup is not
  needed to complete these fixes.

## Dependency upgrades and support policy

- [ ] Update development tooling separately from runtime packages. Keep Poetry
  until the dedicated uv migration below; do not maintain competing lockfiles.
- [ ] Upgrade PyTorch and timm, the NumPy and numerical/image stack, and the
  geospatial stack in separate batches. Update manifest and lockfile together.
- [ ] Verify checkpoint loading, predictions, masks, metrics, and training after
  each batch. Test Numba with JIT enabled as well as the ordinary disabled-JIT suite.
- [ ] Decide the supported Python floor from package wheel availability and tests.
  Python 3.11 is the proposed minimum; validate 3.12 and newer versions before
  advertising support. Update metadata, workflows, Ruff target, README, and lockfile
  together, and document the last release supporting Python 3.10.
- [ ] Evaluate optional WSI dependencies separately; removing mandatory packages
  requires lazy imports and installation tests for every supported extra.
- [ ] Evaluate the five old Dependabot PRs and close superseded ones after their
  replacements land. Group future weekly dependency and Actions updates.

## Agent guidance and gradual typing

- [x] Add a shared `AGENTS.md`, Claude pointer, contributor guide, PR template,
  and compact commit/review skills. Preserve the Ponytail instructions and adapt
  reusable Python and ML guidance to this repository's actual layout and devices.
- [ ] Upgrade mypy and align Ruff, hooks, configuration, and development commands
  in a separate tooling change. Inventory annotation errors and third-party stubs.
- [ ] Establish one explicit, useful module scope with passing type checks in CI;
  start with tractable numerical helpers or output contracts selected by the audit.
  Increase coverage gradually, preventing new errors in the checked scope.
- [ ] Correct optional tensor fields, model/output types, NumPy array dtypes, and
  public parameter/return types without changing runtime representations or APIs.
  Keep shapes, ranges, and device requirements documented and tested at boundaries.
- [ ] Avoid blanket suppressions, manufactured casts, and a project-wide strict
  switch that cannot pass. Document narrow unavoidable third-party gaps.
- [ ] Package and verify `py.typed` once the supported typing surface and downstream
  consumer checks are ready. Validate type information in the built distributions.

## Poetry to uv migration

Migrate after the dependency audit and a verified baseline, in a dedicated PR
without bundling runtime upgrades, a source-layout move, or a build-backend rewrite.

- [ ] Convert metadata to standard `[project]` fields and development dependency
  groups; preserve package identity, extras, Python bounds, URLs, and wheel contents.
- [ ] Generate `uv.lock`, inspect dependency differences, and verify the supported
  CPU/platform matrix. Make PyTorch index choices explicit where needed.
- [ ] Replace Poetry commands in CI, release checks, hooks, contributor and agent
  guidance together. Teach the version guard to read the new metadata location.
- [ ] Verify clean wheel/source installation, public/checkpoint behavior, and the
  prediction baseline. Keep the current build backend unless changing it is needed.
- [ ] Remove `poetry.lock` and obsolete tooling only when uv is the single verified
  project workflow; use locked synchronization in CI and documented local setup.

## Issues and documentation

Prioritize installation issue 68 and metric-matching issue 73, then the working
training example requested in issue 76, model/backbone guidance in issue 74, and
in-memory inference guidance in issue 69. Add reproducing tests for correctness
fixes before changing behavior. Treat 3D support and expanded model architectures
as separate feature work.

- [ ] Make the quick start executable on CPU with an explicit image path, device,
  checkpoint, normalization, and interpretation of outputs. Document GPU use.
- [ ] Provide a runnable training script and update notebooks to the same API.
- [ ] Document supported backbones, checkpoints, extras, ONNX limitations, and
  compatibility changes. Smoke-test examples in CI.
- [ ] Review issue 78 without silently changing preprocessing or dependency
  licensing. Prefer the project's existing normalization helper where suitable.
- [x] Add contributor instructions and repeatable release steps.

## Release and rollback

- [ ] Produce a release candidate and install its wheel and source distribution
  from a clean environment. Test the supported platform and Python matrix.
- [ ] Verify version/tag agreement and write migration notes for intentional breaks.
- [x] Prepare the publishing job for PyPI OIDC, with a `pypi` environment and
  `id-token: write` limited to publication. Remove the stored PyPI token input.
- [x] The maintainer reports configuring the matching PyPI trusted publisher and
  GitHub `pypi` environment as described in [the release guide](releasing.md).
- [ ] Verify publisher authentication on an authorized release. Repository CI
  cannot confirm PyPI account configuration; there is no stored-token fallback.
- [ ] Publish the validated artifacts only after required checks pass and publisher
  configuration is ready. Workflow lint and PR tests cannot verify PyPI account setup.
- [ ] Keep each maintenance batch independently revertible. Preserve the previous
  release and baseline artifacts; fix or revert regressions before advancing.

## Progress and remaining evidence

CI/publication guidance, agent instructions, and Google-style library docstrings
are implemented through PRs 84, 85, and 87. All nine merged-main checks passed;
no package has been published as part of this maintenance work.

The corrective ONNX branch passed 32 local exporter tests with zero skips on
Python 3.11 / PyTorch 2.7.1. Its locked ordinary suite passed 2,041 tests with
5,191 skips: 5,184 gated decoder cases, one CUDA case, four optional runtime
comparisons exercised separately, and two pretrained-image cases. Configured
hooks and workflow lint passed. Both Linux exporter versions passed in
[PR validation run 37964614158](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37964614158)
and in the merged-main run linked above. Half/double tests cover sample
construction; accelerator and end-to-end reduced-precision runtime support
remain unverified.

[The baseline metadata](validation/onnx-baseline.json) records two immutable
checkpoint revisions/digests, one image digest, preprocessing, package versions,
instance counts, and measured ONNX differences. Both models produced identical
instance/type masks on that image, but StarDist dense outputs exceeded their
original tolerance. This is a partial local CPU record, not a representative
baseline or a completed public download/checkpoint-loading compatibility check.

The refreshed GitHub snapshot has 49 open alerts: 27 high, 19 medium, and 3 low,
across 16 normalized package names in `poetry.lock` (GitHub lists Pillow with two
capitalizations). [The review record](maintenance-review.md) preserves the initial
priority groups and reachable paths. No alerts have been dismissed or runtime
dependencies upgraded during this maintenance work; refresh advisory details
before selecting each upgrade.

The direct-import inventory and clean WSI import failure are confirmed, but
required dependency declarations, optional boundaries, representative inference
coverage, and installation issue 68 remain incomplete. Tooling/type checking,
newer Python CI, support-floor changes, and uv remain planned work rather than
features already available in this repository.

Merged PR 90 makes Pillow, Hugging Face Hub, safetensors,
pandas, Shapely, and NetworkX explicit runtime requirements. Bounds include the
locked versions and the versions already exercised in isolated range-based
installation checks. Poetry 2.2.1 refreshes lockfile format/metadata while retaining
all 116 package versions and artifact hashes. The installed-package smoke check
now imports the concrete predictor module. Optional dataset/training imports
remain a separate unresolved boundary; the broad dependency-inventory checkbox
stays open until those APIs and declarations are complete.

Issue 68's original Linux/Python 3.12 failure was reproduced at resolution time:
`torch==2.1.1` has no CPython 3.12 wheel. Current metadata no longer pins that
version. This establishes the original cause, not current Python 3.12 installation
or runtime compatibility. Validate current wheel/source installs on 3.12 before
closing the issue or advertising a newer tested Python matrix.

The dataset import follow-up keeps ordinary training dataset exports, checks
optional imports inside constructors,
removes the unused Albumentations dependency from WSI inference transforms,
and corrects the segmenter's transform keyword. A subprocess regression blocks
Albumentations, PyTables, and Matplotlib while checking public imports, tile
sampling, coordinates, custom transforms, construction, and training dependency
errors. Installed-package checks now cover the dataset and WSI segmenter modules.
This does not complete end-to-end WSI validation or training-extra compatibility.

The now-accessible `WsiSegmenter.segment()` still uses a nonexistent
`self.inferer.device` and passes obsolete `dst`/`maptype` arguments to
`BaseModelInst.post_process`. Repair those concrete API mismatches with a small
CPU segmentation regression during the inference-baseline work before claiming
full WSI segmentation coverage. Slide-backend and training dependency extras,
current Python 3.12 installation, and representative checkpoint tests remain open.
