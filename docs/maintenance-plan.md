# Repository maintenance plan

Refresh the project in small, independently reviewable changes. Establish reliable
CI and reproducible predictions before upgrading runtime dependencies, so existing
users retain working APIs, checkpoints, and segmentation results.

## Current baseline

Status reconciled on 2026-10-12 against main at `bd69d4e`, version `0.1.30`.
PRs 84/85/87 provide CI, agent guidance, Google docstrings, and corrected ONNX
exports. PRs 89/90/91 repair WSI/optional dataset imports and declare direct
runtime requirements. PR 92 adds Python 3.12 source and clean-install checks.
PRs 93/94 update the HTTP/cache stack and Pillow. PR 86 is superseded and closed.
PR 96 updates geometry/serialization, PR 97 modernizes developer tooling and
adds the first strict typing gate, and PR 98 repairs CPU WSI tile execution.
PR 99 preserves repo-only checkpoint references and removes the one-off docs
reports. PR 100 establishes the sampled CPU inference comparison policy and
cleanup triggers, and retires temporary diagnostics. Comparisons remain manual.
PR 101 calibrates pretrained StarDist ONNX classifier precision independently of
the exporter and requires exact dense class decisions and final masks.
All 13 hosted checks passed for PRs 97/98/99/100/101.
The old ONNX review checkout is historical; its corrections are merged.

The completed upgrade PRs passed all 12 hosted checks on Python 3.10/3.11/3.12,
including source suites, build, six clean installs and two ONNX jobs. See
[Python 3.12 validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38041808171),
[HTTP validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38044641492),
and [Pillow validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38045922826).
The Python minimum remains 3.10. The model dependency batch is now in progress;
the numerical stack remains unchanged. No maintenance release is published.

The original contributor PRs passed ordinary tests but skipped runtime comparisons
without optional ONNX packages. They were merged before the corrective patch;
PR 87 supplied the batching/state/dtype fixes and explicit runtime checks. This
sequence deviated from the original merge gate and is now corrected on main.
The pretrained ONNX patch calibrates StarDist class-logit precision separately
from physical outputs and requires exact dense class decisions and final masks.
Representative prediction coverage remains open; passing these sampled checks
does not establish dataset accuracy or accelerator compatibility.

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
   (93/94); geometry/serialization is complete (96). Developer tooling is complete (97). Refresh advisory triage per batch.
   Record affected paths and resolved advisory IDs; update manifest/lock together
   where needed. These focused fixes can precede the broader numerical baseline,
   with image-decoding/checkpoint checks appropriate to each changed dependency.
4. Complete the reproducible prediction baseline: immutable checkpoint/image
   identities, encoder/class/output settings, preprocessing, dense outputs,
   reconstructed masks and counts. PR 101 resolves the sampled StarDist ONNX
   comparison using independent native precision calibration. For each incoming
   batch, establish the affected image/model checks before comparing versions;
   broader dataset and accelerator validation remain release work.
5. Developer tooling and the strict release-tool typing scope are complete (97).
   Expand library typing separately from runtime upgrades and the uv migration.
6. Upgrade model and numerical/image dependencies in separate batches once their
   baseline checks exist; the geometry/serialization batch is complete (96).
   Add newer Python CI coverage after dependency wheel/compatibility checks;
   change the Python floor only in an explicit support
   policy patch. The Python floor remains 3.10. Python 3.12 source and clean
   distribution checks are already merged; newer versions require validation.
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
- [x] Record immutable checkpoint revisions/checksums, encoder, numeric class IDs,
  normalization, image identity, and output settings for the initial two models.
  Broader model coverage and verified semantic class names remain open.
- [ ] Capture representative dense predictions, reconstructed instance masks,
  and instance counts. Define tolerances before comparing dependency changes.
- [x] Exercise pinned public checkpoint downloads and safetensors loading for the
  initial CellPose/StarDist fixtures. Registered-name loading remains open.
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
- [x] Repeat CPU pretrained-image comparisons with the recorded checkpoint/image
  on PyTorch 2.7.1 and 2.14.1. ONNX covers dense prediction; reconstruction remains
  in Python. Broader images/devices remain separately scoped.
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
- [ ] Finish the current PyTorch/timm batch with the six-image comparisons,
  source, minimum-version, ONNX, and installed-package checks described below.
- [ ] Upgrade the broader numerical stack with its own affected-path baseline.
  Update manifest and lockfile together.
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


## Cleanup after maintenance work

Review temporary material after each completed batch and before the maintenance
release candidate. Record its purpose and removal trigger in the relevant plan
item or PR; use this plan rather than creating another inventory document.

- Remove completed investigation notes, diagnostics, duplicate reports, and
  obsolete phase plans once their remaining facts have reached user/contributor
  guidance or a maintained regression. PRs and Git history retain past evidence.
- Keep fixtures only while a maintained check consumes them. Remove superseded
  data with its obsolete check, and keep reference data outside distributions.
- Review CI duration, downloads, matrix versions, and artifact storage. Retire
  temporary jobs and redundant checks; use manual upgrade/release runs for costly
  investigations without a continuing per-PR purpose. Preserve distinct supported
  source, installed-package, export, and release contracts.
- Revisit deferred design drafts when adopted or cancelled. Consolidate the
  maintenance plan/review history at the end, retaining actionable open work and
  current compatibility guidance instead of a permanent progress diary.

Release cleanup checks:

- [ ] Remove resolved maintenance notes and unused diagnostic scripts/fixtures.
- [ ] Inspect wheel/source contents and sizes for accidental test/report payloads.
- [ ] Review CI timings and retire obsolete jobs, matrix entries, and reports.
- [ ] Check remaining documentation links and unresolved tasks after consolidation.

## Remaining baseline evidence

[The maintained baseline fixtures](../tests/fixtures/checkpoint_baseline/README.md)
record two immutable checkpoint revisions/digests, one image digest, preprocessing,
reference versions, preserved PyTorch arrays, and instance counts. Measured ONNX
differences remain in [the historical review](maintenance-review.md); they are
not part of these PyTorch fixtures. Native float32/float64 calibration over six
existing images found StarDist class-logit error requiring an absolute allowance
above 1.35e-4 at rtol=1e-4, independently of ONNX. The pretrained StarDist test
uses the existing CellPose class-logit budget (atol=2e-4); foreground/ray bounds
stay at 1e-5. Raw values, shapes, dtypes, dense class decisions, and final masks
remain checked. The pinned real-image checks pass on both exporter versions.

Ordinary CI runs only the offline fixture integrity check. Run the optional
checkpoint comparisons manually for numerical upgrades and releases; they are
scoped to the recorded image and checkpoints. A one-off Linux/macOS investigation
confirmed identical object partitions and class maps across the sampled CPU
backends. Compare class probabilities using the existing bounds, raw physical
heads using their existing bounds, and exact instance partitions/background with
positive-ID bijections. No production inference, reference arrays, or thresholds
change in the CPU baseline. Broader image/model coverage remains open; sampled
raw pretrained ONNX parity now uses the separately calibrated class-head budget.
For the model dependency batch, compare the old and proposed environments on
both pinned checkpoints and six existing images, with unchanged dense bounds,
exact dense class decisions, object partitions, and final type maps. Also run
the frozen fixture comparisons, source suite, CPU forward/backward checks, and
ONNX tests at the supported minimum/current versions. Keep numerical dependency
upgrades separate until their affected post-processing and metric checks exist.
This batch-specific evidence does not complete broader dataset/GPU validation.
The repo-level fixtures and integration test are excluded from distributions
by living outside the runtime package. This covers one image and two models,
not a representative dataset, registered-name loading, or accelerator validation.

The ordinary suite and installed-package checks do not establish GPU/AMP,
representative-image, full-slide, training-extra, or newer-Python coverage.
Issue 68 still needs verification against a published release. Security advisory
counts are dated snapshots; remaining alerts require applicability triage.
Completed batch validation is recorded in the relevant PRs and
[the review record](maintenance-review.md), rather than repeated in this plan.

Current model batch: raise the PyTorch security floor to 2.13.0 and lock
2.14.1/torchvision 0.29.1 with timm 1.0.30. Raise fsspec's floor to 2026.6.0
and lock 2026.9.0; require setuptools 83.0.0 or newer and lock 84.0.0. This addresses
GHSA-rrmf-rvhw-rf47, GHSA-qfhq-4f3w-5fph,
GHSA-vgrw-7cvw-pwgx, GHSA-887c-mr87-cxwp, GHSA-27vj-qcqg-25rc, and
GHSA-h35f-9h28-mq5c. The library scripts its fixed Swish/Mish functions with
torch.jit.script; it does not call the affected LSTM, sequence-unpacking,
CTC-loss, or fsspec reference-filesystem APIs. The setuptools manifest issue
does not affect our Poetry build.
Update the existing ONNX minimum job to 2.13.0/torchvision 0.28.0. PyTorch's
new Linux CUDA dependencies require separate driver/device verification before
release; CPU evidence does not validate CUDA/AMP. Temporary six-image reference
arrays and the probe stay outside the repository and retire after this batch;
the PR retains the results, with no additional CI job or distribution payload.

## Deferred structure and validation records

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and feature isolation. It is deferred structural work, not a prerequisite refactor
bundled into dependency upgrades. The GPU proposal is preserved on
`docs/gpu-postprocessing-proposal` and remains deferred after maintenance.
