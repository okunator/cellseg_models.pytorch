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

- [ ] Verify the locked environment on the existing Python 3.10 and 3.11 matrix.
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

- [ ] Review PRs 80 and 81 individually, preserving contributor attribution.
- [ ] Install ONNX, ONNX Script, and ONNX Runtime in an explicit validation job.
  Require that export and runtime comparison tests execute rather than skip.
- [ ] Test batches of size one and greater than one, output names, fixed spatial
  dimensions, missing dependencies, export failure, model dtype, and restoration
  of model training state, including mixed child-module states.
- [ ] Repeat pretrained-image comparisons using the recorded checkpoint and image.
  Document that ONNX covers dense prediction; reconstruction remains in Python.
- [ ] Check the declared minimum export PyTorch version and a current supported
  version. Keep export dependencies optional for ordinary PyTorch inference.
- [ ] Merge separately only after review and passing checks, then validate their
  combined behavior. Resolve duplicated helpers if needed without broad refactoring.

## Dependency upgrades and support policy

- [ ] Update development tooling separately from runtime packages; keep Poetry.
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
- [ ] Publish the validated artifacts only after required checks pass. Consider
  PyPI trusted publishing after configuring the publisher account separately.
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

Hosted Linux checks on Python 3.10 and 3.11 remain required before merging the
first patch. Checkboxes above identify implemented changes; the stage is complete
only when hosted validation passes. No contribution has been merged and no package
has been published.
