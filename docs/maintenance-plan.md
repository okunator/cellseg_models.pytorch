# Repository maintenance plan

Refresh the project in small, independently reviewable changes. Establish reliable
CI and reproducible predictions before upgrading runtime dependencies, so existing
users retain working APIs, checkpoints, and segmentation results.

## Current baseline

Status reconciled on 2026-10-10 against main at `edfc92c`, version `0.1.30`.
PRs 84/85/87 provide CI, agent guidance, Google docstrings, and corrected ONNX
exports. PRs 89/90/91 repair WSI/optional dataset imports and declare direct
runtime requirements. PR 92 adds Python 3.12 source and clean-install checks.
PRs 93/94 update the HTTP/cache stack and Pillow. PR 86 is superseded and closed.
The old ONNX review checkout is historical; its corrections are merged.

The completed upgrade PRs passed all 12 hosted checks on Python 3.10/3.11/3.12,
including source suites, build, six clean installs and two ONNX jobs. See
[Python 3.12 validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38041808171),
[HTTP validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38044641492),
and [Pillow validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38045922826).
The Python minimum remains 3.10. Model/numerical locked versions are retained;
ONNX export requires PyTorch 2.7 or newer. No maintenance release is published.

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
2. Direct declarations and dataset import repairs are merged (90/91). Python
   3.12 source and clean-install checks pass (92). Finish the remaining optional
   feature inventory and test real slide/training backends; issue 68 still needs
   the relevant fixes available in a published release.
3. Continue grouped dependency updates: HTTP/cache and Pillow are complete
   (93/94); geometry/serialization is complete (96). Developer tooling is next. Refresh advisory triage per batch.
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
   policy patch. The Python floor remains 3.10. A separate CI patch adds Python 3.12 source
   and clean distribution checks; newer versions still require validation.
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
- [x] Declare the six audited direct runtime imports explicitly (PR 90).
- [x] Repair optional training dataset imports and WSI visualization imports
  without dynamic package export workarounds (PRs 89/91).
- [ ] Complete optional feature declarations and independently tested import
  boundaries across slide, training and export APIs.
- [x] Reproduce issue 68's old PyTorch pin failure and validate current locked
  source and range-based wheel/source installs on Linux Python 3.12 (PR 92).
- [ ] Verify the published release resolves the reported installation path before
  closing issue 68; other platforms and optional integrations remain open.
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

- [x] Update developer tools independently, with Ruff as the single formatter/
  linter and a narrow strict release-tool gate. Poetry remains authoritative.
- [x] Upgrade HTTP/cache security dependencies and Pillow independently with
  compatibility evidence (PRs 93/94).
- [x] Upgrade geometry/serialization with round-trip, cross-version, coordinate,
  and JIT-enabled geometry checks (PR 96).
- [ ] Upgrade PyTorch/timm and the broader numerical stack after the reproducible
  prediction baseline. Update manifest and lockfile together.
- [ ] Verify checkpoint loading, predictions, masks, metrics, and training after
  each batch. Test Numba with JIT enabled as well as the ordinary disabled-JIT suite.
- [ ] Decide the supported Python floor from package wheel availability and tests.
  Python 3.12 is tested; a 3.11 minimum is only a proposal. Validate newer
  versions before
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
- [x] Upgrade mypy and align Ruff, hooks, configuration, and developer commands.
- [x] Establish a first strict CI scope for the release-version guard. This checks
  developer tooling, not the segmentation library.
- [ ] Inventory library annotation errors/stubs and add a tractable numerical
  helper or output-contract scope, then increase coverage gradually.
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

Security counts are dated snapshots, not live completion criteria. HTTP/cache
and Pillow updates remediate the recorded affected version ranges; remaining
advisories still require applicability triage and updates. No alerts were dismissed.

Direct declarations, dataset import repairs and Linux Python 3.12 coverage are
complete. Representative inference, full WSI execution, other optional packages,
newer Python versions, tooling/type checking, and uv remain incomplete.

Merged PR 90 makes Pillow, Hugging Face Hub, safetensors,
pandas, Shapely, and NetworkX explicit runtime requirements. Bounds include the
locked versions and the versions already exercised in isolated range-based
installation checks. Poetry 2.2.1 refreshes lockfile format/metadata while retaining
all 116 package versions and artifact hashes. The installed-package smoke check
now imports the concrete predictor module. The dataset import repair is merged in PR 91; broader optional feature coverage
remains incomplete; the broad dependency-inventory checkbox
stays open until those APIs and declarations are complete.

Issue 68's original Linux/Python 3.12 failure was reproduced at resolution time:
`torch==2.1.1` has no CPython 3.12 wheel. Current metadata no longer pins that
version. PR 92 subsequently validated current Linux Python 3.12 source and wheel/source
installs. Published release verification is still required before closing the issue.

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
other-platform installation and representative checkpoint tests remain open.


## Python 3.12 installation validation

The Python 3.12 CI patch extends both the locked source matrix and the clean
wheel/source install matrix, preserving Python 3.10 and 3.11 checks. It exercises
current repository artifacts on Linux, covering the environment family reported
in issue 68. The historical `torch==2.1.1` requirement cannot resolve on CPython
3.12; the current manifest has no such exact pin. This patch changes neither
runtime dependencies nor the lockfile nor the Python minimum.

All 12 hosted checks passed for PR 92, including the new source/install jobs. It does not
validate every optional package, GPU, platform, or Python version above 3.12.
Issue 68 should remain open until the relevant fixes are available in a published
release and the maintainer decides whether its original report is resolved.


## Download and cache security batch

After merged PRs 91 and 92, the source and clean wheel/source installation matrix
covers Python 3.10, 3.11, and 3.12 on Linux. Merged PR 93 raises
Requests to at least 2.33.0, urllib3 to 2.8.0, IDNA to 3.15, and filelock to 3.20.3.
Explicit transitive security floors prevent published installs from accepting
older affected versions even when another package has already installed them.
Only these four locked versions change; the model and numerical stack is retained.

[The review record](maintenance-review.md) lists affected advisory IDs and their
preconditions. Hermetic localhost tests cover chunked streaming, gzip decoding,
and redirects. Separate checks cover lock exclusion/release, Unicode hostname
encoding, and immutable checkpoint cache digests. These checks establish basic
compatibility, not exploitation of every advisory or broad prediction parity.


## Pillow security batch

Merged PR 94 raises the published Pillow minimum and lockfile to 12.3.0,
remediating 18
recorded advisory ranges without changing other locked packages.
[The validation record](validation/pillow-upgrade.json) captures 46 identical
preprocessing/codec/annotation arrays and unchanged dense predictions and masks
for two immutable checkpoints on the recorded image. Representative inference
and the StarDist ONNX gap remain open. All 12 hosted checks passed.
PyArrow/geospatial serialization is the next grouped batch.


## Deferred structure and validation records

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and feature isolation. It is deferred structural work, not a prerequisite refactor
bundled into dependency upgrades. The GPU proposal is preserved on
`docs/gpu-postprocessing-proposal` and remains deferred after maintenance.

Retain the existing ONNX/Pillow JSON records until checkpoint identities,
preprocessing and expected behavior have maintained baseline fixtures. Then
consolidate reusable data and remove one-off reports from docs, preserving
historical evidence in PR descriptions and Git history. Add no more one-off
upgrade JSON reports; keep validation in tests and the relevant PR descriptions.


## Developer toolchain batch

Update pytest/cov/xdist, pre-commit/virtualenv, mypy, Ruff, Matplotlib/fontTools,
and scriv in the development group. Remove unused Black/isort/Flake8 toolchains
and their dependencies. Runtime package versions and artifact hashes stay intact.
FontTools selects a compatible version by Python version while retaining 3.10.

Ruff keeps the effective E4/E7/E9/F rule scope; remove the overridden top-level
rule list and use current nested configuration. Hooks and the project pin share
Ruff 0.17.0. Strict mypy initially checks only the release-version guard; its
manifest version validation now explicitly requires a string, with a regression.
A developer-only CI job runs these checks without installing the model stack.
Library annotation inventory, numerical type coverage, and full-repo lint cleanup
remain separate work. No general ignore policy or manufactured casts are added.
