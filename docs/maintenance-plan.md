# Repository maintenance plan

Refresh the project in small, independently reviewable changes. Establish reliable
CI and reproducible predictions before upgrading runtime dependencies, so existing
users retain working APIs, checkpoints, and segmentation results.

## Current baseline

Status reconciled on 2026-10-12 against main at `c870c5e`, version `0.1.30`.
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
PR 102 updates PyTorch/timm and their security floors, with six-image CPU
comparisons and corrected torchvision/Python metadata. All 13 hosted checks
passed for PRs 97/98/99/100/101/102/103/104/105/106/107/108/109. PR 103 completes the
numerical/image batch with compiled-kernel coverage and NumPy 2 regressions.
PR 104 fixes shared percentile clipping/zero spans and DRFNS byte saturation.
PR 106 fixes low-IoU instance assignment, FP/FN axes and sparse IDs, with
independent metric validation and public regressions.
PR 109 repairs HDF5 sample counts and auxiliary target association.
PR 110 completes the uv migration with all 13 hosted checks passing.
PR 107 retires legacy notebooks; PR 108 preserves the README quick start.
Replacement documentation and examples are deferred to the docs-pages phase.
PR 105 updates locked HTML/highlighting transitives and removes unnecessary
runtime floors; downloader/cache protections remain. The old ONNX review checkout
is historical; its corrections are merged.

The completed upgrade PRs passed all 12 hosted checks on Python 3.10/3.11/3.12,
including source suites, build, six clean installs and two ONNX jobs. See
[Python 3.12 validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38041808171),
[HTTP validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38044641492),
and [Pillow validation](https://github.com/okunator/cellseg_models.pytorch/actions/runs/38045922826).
The model batch is merged. The Python minimum remains 3.10, with 3.14.1
excluded by torchvision. The numerical/image batch is merged. No
maintenance release is published.

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
   (93/94); geometry/serialization (96), developer tooling (97), and the model
   stack (102) and numerical/image stack (103) are complete. Refresh advisory triage per batch.
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
6. Metric-matching issue 73 is resolved on main (106), legacy notebooks are
   retired (107/108), and real HDF5 dataset repairs are merged (109). The bounded
   dependency inventory is complete for this migration; further optional/backend
   repairs are separate follow-ups rather than an open-ended prerequisite to uv.
   User guides and examples move to the docs-pages phase.
   The shared percentile normalizer and DRFNS fixes are merged (104). Model,
   numerical/image, and geometry batches are merged (102/103/96).
   Add newer Python CI coverage after dependency wheel/compatibility checks;
   change the Python floor only in an explicit support
   policy patch. The Python floor remains 3.10. Python 3.12 source and clean
   distribution checks are already merged; newer versions require validation.
7. The uv migration is merged (110). Add a small library typing scope, then
   perform cleanup/release-candidate checks. Keep API docstrings and release notes with
   functional changes; build user documentation pages and examples in their own
   later phase. Keep trusted publisher authentication pending until an
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

- [x] Triage the currently open dependency advisories by reachable paths.
  PR 105 is merged with all 13 hosted checks passing. GitHub rescanning closes
  the remaining alerts; no open dependency alerts were reported on 2026-10-12.
  Refresh the audit before the release candidate; this is not a blanket security
  certification or proof against future advisories.
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
  linter and a narrow strict release-tool gate (97). The current uv migration
  supersedes its Poetry workflow while preserving the same tool versions.
- [x] Upgrade HTTP/cache security dependencies and Pillow independently with
  compatibility evidence (PRs 93/94).
- [x] Upgrade geometry/serialization with round-trip, cross-version, coordinate,
  and JIT-enabled geometry checks (PR 96).
- [x] Upgrade PyTorch/timm with six-image CPU comparisons, minimum/current ONNX,
  and source/clean-package validation on Python 3.10/3.11/3.12 (PR 102).
  GPU/AMP and driver verification remain release work.
- [x] Upgrade the numerical/image stack with affected-path CPU comparisons,
  NumPy compatibility regressions and JIT coverage; Python 3.10/3.11/3.12 source
  and clean-package checks pass (PR 103).
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
- [ ] Add the first strict library scope: `utils/convolve.py`, shared by inference,
  SSIM and spatially weighted losses. The initial inventory finds five annotation
  errors: padding accepts a sequence, and device/dtype defaults are optional.
  Correct annotations without changing numerical behavior; verify kernel values,
  padding, channel isolation and gradients. Reuse the existing Python 3.11 source
  job; leave the developer-only environment free of runtime dependencies.
  Broader annotation/stub inventory remains incremental follow-up work.
- [ ] Correct optional tensor fields, model/output types, NumPy array dtypes, and
  public parameter/return types without changing runtime representations or APIs.
  Keep shapes, ranges, and device requirements documented and tested at boundaries.
- [ ] Avoid blanket suppressions, manufactured casts, and a project-wide strict
  switch that cannot pass. Document narrow unavoidable third-party gaps.
- [ ] Package and verify `py.typed` once the supported typing surface and downstream
  consumer checks are ready. Validate type information in the built distributions.

## Poetry to uv migration

The migration is merged in PR 110 after the bounded dependency audit and sampled
baseline checks. Standard project metadata, uv 0.11.4 and `uv.lock`
replace the Poetry CLI workflow. The build backend remains poetry-core 2.2.1,
now explicitly pinned. All 128 active Poetry package/version records are retained;
only one unreachable lock entry is retired and an equivalent CUDA toolkit version
is normalized. Further dependency upgrades remain separate.

Completion requires the existing hosted gates and installed-artifact checks;
no new CI job or speculative extra is introduced.

- [x] Convert remaining metadata/dependencies to standard `[project]` fields and
  development dependency groups (name/Python bounds are already standard);
  preserve package identity, extras, Python bounds, URLs, and wheel contents.
- [x] Generate `uv.lock` preserving active resolved versions. Keep PyPI as the
  project source and the two CPU PyTorch overrides explicit in the existing ONNX
  jobs. Supported Python/Linux CI verification remains the migration gate.
- [x] Replace Poetry commands in CI, release checks, hooks, contributor and agent
  guidance together. Teach the version guard to read the new metadata location.
- [x] Verify clean wheel/source installation, public/checkpoint behavior, and the
  sampled prediction baseline (110). All 13 hosted gates pass; local ordinary
  tests, compiled helpers and the two pinned checkpoint comparisons pass.
  Keep the current build backend unless changing it is needed.
- [x] Remove `poetry.lock` and obsolete tooling now that uv is the single locally
  verified project workflow; use locked synchronization in CI and documented local setup.

## Issues and documentation

Installation issue 68 still needs a published-release check; metric issue 73 is
fixed on main. Continue concrete correctness and dependency/import work during
maintenance. Training/fine-tuning examples (76), model/backbone guidance (74),
in-memory inference guidance (69) and the quick start belong to the later
user-documentation pages phase. Treat 3D support and expanded architectures as
separate feature work.

- [x] Fix `percentile_normalize99` clipping/zero spans and DRFNS byte saturation
  with value, input-preservation and strict caller regressions (PR 104). Hosted
  source, compiled-kernel and clean-package checks pass. Clipping can intentionally
  change enhanced post-processing outputs, as documented in the release notes.
- [x] Merge the issue 73 correction (PR 106): unique low-IoU matches, correct
  FP/FN axes, sparse IDs and explicit precision semantics. All 13 hosted checks
  pass; default-threshold values and AJI matching policy remain unchanged.
  The issue comment is drafted for maintainer review; release availability remains
  pending publication.
- [x] Retire the legacy example notebooks (107) and preserve the README (108).
  The proposed training example is deferred; Git history preserves it without
  an archived copy in the repo.
- [x] Merge HDF5 training dataset repairs (109) found by the real optional-backend
  audit: configured-image sample counts, aligned selected arrays and correct
  auxiliary target ordering across instance keys/transforms. Validate real
  Array/CArray/EArray files; keep dependency declarations separate.
- [ ] Build documentation pages after the maintenance work, with a verified CPU
  quick start, training and fine-tuning examples, model/checkpoint/backbone and
  optional-dependency guidance. Validate examples when introduced and decide
  the smallest appropriate CI coverage then; add no speculative jobs now.
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
PR 103 compares six images, two pinned models, three annotation masks, training
targets, metrics, tissue masks and small reconstruction fixtures before/after
numerical updates, separately with Numba enabled and disabled. Keep those
batch results in the PR; future changes repeat only the affected checks.
PR 104 preserves sampled unclipped results and the maintained checkpoint
comparisons while fixing requested clipping/zero spans and DRFNS failures.
Keep its direct regressions; the validation record lives in the PR.
The repo-level fixtures and integration test are excluded from distributions
by living outside the runtime package. This covers one image and two models,
not a representative dataset, registered-name loading, or accelerator validation.

The ordinary suite and installed-package checks do not establish GPU/AMP,
representative-image, full-slide, training-extra, or newer-Python coverage.
Issue 68 still needs verification against a published release. Security advisory
counts are dated snapshots; remaining alerts require applicability triage.
Completed batch validation is recorded in the relevant PRs and
[the review record](maintenance-review.md), rather than repeated in this plan.

Completed model and numerical investigations are retained in
[PR 102](https://github.com/okunator/cellseg_models.pytorch/pull/102) and
[PR 103](https://github.com/okunator/cellseg_models.pytorch/pull/103), with their
temporary references retired after use. OpenCV 4 and Python 3.10 remain supported;
OpenCV 5 and newer Python coverage require separate compatibility checks.
PR 105 locks Soup Sieve 2.10 and Pygments 2.21.0 and removes their runtime
declarations and the earlier fsspec/setuptools advisory floors, retaining patched
lock versions. Their vulnerable selector, ADL lexer,
reference-template and setuptools build paths are not used by this library.
Published installations follow parent requirements; they do not consume this lock.

Prefer updating the dependency that owns a transitive requirement. Retain a
transitive security floor only for a reachable vulnerable path that the parent's
minimum does not protect. Requests still permits older urllib3/IDNA on downloader
traffic, and PyTorch/Hub permit older filelock on checkpoint caching, including
Hub's soft-lock fallback. Keep these three floors until parent minimums protect
those paths. Current Requests, PyTorch and Mpire releases do not tighten the
relevant minimums. Hub's transport migration and libpysal's newer Python minimum
need separate compatibility checks before further parent upgrades.

Batch validation is recorded in [PR 105](https://github.com/okunator/cellseg_models.pytorch/pull/105).
Its temporary probes remain outside the repository and retire after the batch.
GitHub rescanning reports no remaining open alerts on 2026-10-12. Refresh
the audit before the release candidate.

## Deferred structure and validation records

[The module-boundary draft](module-boundaries-draft.md) proposes one-way imports
and feature isolation. It is deferred structural work, not a prerequisite refactor
bundled into dependency upgrades. The GPU proposal is preserved on
`docs/gpu-postprocessing-proposal` and remains deferred after maintenance.
