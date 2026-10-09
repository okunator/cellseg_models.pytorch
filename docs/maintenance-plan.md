# Repository maintenance plan

Refresh the project in small, independently reviewable changes. Establish reliable
CI and reproducible predictions before upgrading runtime dependencies, so existing
users retain working APIs, checkpoints, and segmentation results.

## Current baseline

The starting commit is `8deead255420b4a581e45d12fa471478d0509b42`, version
`0.1.30`. The inspected runs for [CellPose ONNX PR 80](https://github.com/okunator/cellseg_models.pytorch/pull/80)
and [StarDist ONNX PR 81](https://github.com/okunator/cellseg_models.pytorch/pull/81)
passed their test steps. Their Python 3.10 runs each reported 2,012 passed and
5,187 skipped. Codecov uploads failed with HTTP 429 and an empty token; fail-fast
cancelled the Python 3.11 jobs during cleanup. The ONNX Runtime comparisons skipped
because ONNX was absent. These results do not establish export correctness.

Of the skipped cases, 5,184 belong to one deliberately gated decoder matrix.
The `--slow`, `--optional`, and `--cuda` flags select only their own categories;
run these categories separately from ordinary tests.

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
- [ ] Test batches of size one and greater than one, output names, fixed spatial
  dimensions, missing dependencies, export failure, model dtype, and restoration
  of model training state, including mixed child-module states.
- [ ] Repeat pretrained-image comparisons using the recorded checkpoint and image.
  Document that ONNX covers dense prediction; reconstruction remains in Python.
- [x] Check the declared minimum export PyTorch version and a current supported
  version. Keep export dependencies optional for ordinary PyTorch inference.
- [ ] Merge separately only after review and passing checks, then validate their
  combined behavior. Resolve duplicated helpers if needed without broad refactoring.

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
- [ ] Add contributor instructions and repeatable release steps.

## Release and rollback

- [ ] Produce a release candidate and install its wheel and source distribution
  from a clean environment. Test the supported platform and Python matrix.
- [ ] Verify version/tag agreement and write migration notes for intentional breaks.
- [x] Prepare the publishing job for PyPI OIDC, with a `pypi` environment and
  `id-token: write` limited to publication. Remove the stored PyPI token input.
- [ ] The maintainer configures the matching PyPI trusted publisher and GitHub
  environment as described in [the release guide](releasing.md), then verifies
  authentication on an authorized release. There is no stored-token fallback.
- [ ] Publish the validated artifacts only after required checks pass and publisher
  configuration is ready. Workflow lint and PR tests cannot verify PyPI account setup.
- [ ] Keep each maintenance batch independently revertible. Preserve the previous
  release and baseline artifacts; fix or revert regressions before advancing.

## Progress

The first patch implements the CI and publication changes above. Local validation
on macOS with Python 3.11 passed 2,013 tests; 5,185 cases remained intentionally
skipped. Ten model unit tests now disable pretrained encoder downloads, and the
ordinary source suite runs with Hugging Face Hub offline. The five release guard
tests passed, as did workflow validation and strict distribution metadata checks.
Clean wheel and source installations passed dependency checks, public imports,
and a CPU optimizer step outside the checkout, using the dependencies selected
from the declared ranges rather than the development lockfile.

All seven hosted checks for the initial implementation passed on
[run 37948669248](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37948669248),
including Linux source tests and clean wheel/source installs on Python 3.10 and 3.11.
All seven hosted checks also passed for the completed agent guidance and OIDC
publishing configuration on [run 37950270241](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37950270241).
Checkboxes identify implemented changes; publisher authentication
still requires an authorized release to verify authentication. Maintenance PRs
84 and 85 and ONNX contributions 80 and 81 have since been merged. No package
has been published as part of this maintenance work.

The contributor ONNX runtime tests were exercised with the optional packages
installed. Both original PRs fail dynamic-batch parity because batch-one tracing
specializes the graph. Corrections and additional tests were prepared independently
on `fix/onnx-maintenance`. The original contributions were merged without those
corrections; the fixes and explicit ONNX CI gate are now carried separately on
`fix/onnx-export-correctness`, based on docstring PR 86. The combined source suite passed 2,041 tests before the export-version
guard was tightened. Real-checkpoint comparisons produced identical instance and
type masks on the recorded image, but StarDist still exceeds the original dense
tolerance. PyTorch 2.5 export failed, so the prepared export API requires the tested
minimum 2.7. All nine hosted checks passed on [run 37955000998](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37955000998),
including explicit ONNX tests on PyTorch 2.7.1 and 2.14.1. StarDist dense parity
and a representative prediction baseline remain open before upgrading runtime
dependencies.

The dependency audit snapshot contains 42 open alerts across 15 packages.
[The review record](maintenance-review.md) identifies the initial priority groups
and reachable paths; no alerts have been dismissed and no runtime upgrades have
been applied.

The follow-up ONNX fix branch preserves the merged contributor attribution and
Google-style docstrings. Local CPU checks on Python 3.11 / PyTorch 2.7.1 passed
32 exporter tests with zero skips (the two pretrained-image cases were excluded
and retain the separately recorded results). Configured hooks and workflow lint
passed. The locked ordinary suite passed 2,041 tests with 5,191 skips: the gated
decoder matrix, one CUDA case, four optional runtime comparisons (run separately
above), and two pretrained-image cases. Hosted validation of this follow-up
remains required. Runtime dependency
bounds and the lockfile are unchanged; PyTorch 2.7 is required only for ONNX export.
