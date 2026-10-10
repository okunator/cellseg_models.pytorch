# Maintenance review — 2026-10-09

## CI baseline

All seven checks passed for maintenance PR 82 at `61a53d9`, including source tests
and clean wheel/source installs on Python 3.10 and 3.11. That initial run is
[37950270241](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37950270241).
The maintenance changes were subsequently merged through PR 84. All nine checks
passed on current main (`574a10d`) in [run 37966314883](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37966314883).
Trusted-publisher authentication requires an authorized release; passing PR checks do not verify PyPI account configuration.

## ONNX contribution findings

Reviewed [CellPose PR 80](https://github.com/okunator/cellseg_models.pytorch/pull/80)
at `46cd9f849cff63707ac801bf8a955b27f7d96fa1` and
[StarDist PR 81](https://github.com/okunator/cellseg_models.pytorch/pull/81)
at `590d8d575abe94f85e5f96adc62d81815cd79f0f`.
Both contributions preserve the existing inference API and keep reconstruction
in Python. Their optional dependencies are imported lazily.

The following defects affected both original exporters and are corrected in PR 87:

| Finding | Concrete trigger and result | Merged correction |
| --- | --- | --- |
| Dynamic batch export specializes batch one | Export with the default batch-one shape and `dynamic_batch=True`; ONNX Runtime rejects batch two with `Expected: 1`. Both original runtime tests fail once optional packages are installed. | Trace with a batch of at least two for dynamic export; compare runtime batches one, two, and three and preserve fixed spatial dimensions. |
| Individual training states are lost | A training model with an evaluation-mode BatchNorm child ends with that child training, even after export fails. Failure while constructing the sample input also leaves the model in evaluation mode. | Construct the sample before changing modes, then restore each module's exact training flag in `finally`. |
| Sample dtype differs from model dtype | A float64 model receives a float32 sample and fails with an input/bias dtype mismatch. The same forced dtype affects half-precision models. | Select the sample device and floating dtype from the model; test float16 and float64 input construction. |
| Advertised minimum export version is unverified and fails | PyTorch 2.5.1 fails ONNX conversion with both ONNX Script 0.7.2 and 0.1.0. This does not prove every historical package combination fails, but the unrestricted installation advice cannot establish support. | Require the tested minimum PyTorch 2.7 for this new export API. Ordinary inference retains its existing PyTorch bounds. |

Corrections were prepared on `fix/onnx-maintenance` and then extracted into
PR 87 after the original contributions were merged. PR 87 is now merged on main,
including Google-style exporter docstrings, regression tests, and the explicit
ONNX gate. PR 86 was closed as superseded. Ordinary inference dependency bounds
and the lockfile remain unchanged. No external contributor review or message
has been posted.

## Validation and remaining gaps

Validation used macOS arm64, Python 3.11, CPU float32, the repository's locked
numerical packages, ONNX 1.23.2, ONNX Script 0.7.2, ONNX Runtime 1.31.0, and
ML Dtypes 0.5.4. The latter preserves compatibility with locked NumPy 1.26.4;
ML Dtypes 0.6.0 requires NumPy 2. No project dependencies or lockfile were changed.

- Original contributions: 8 passed, 2 failed, 2 real-checkpoint tests skipped.
- Prepared batching/state/dtype corrections: 28 targeted tests passed on PyTorch
  2.7.1 and 2.14.1. Real-checkpoint tests ran separately.
- After the minimum-version guard was updated: 32 targeted tests passed on
  PyTorch 2.7.1 and 2.14.1. The four added cases reject unsupported export versions.
- Combined ordinary source suite before that guard change: 2,041 passed and
  5,187 skipped. These skips are the gated decoder matrix, one CUDA test, and the
  two separately exercised real-checkpoint tests.
- Missing optional dependencies, invalid shapes, export failure, input allocation
  failure, mixed child modes, sample dtype, output names, dynamic/fixed batches,
  and fixed spatial dimensions are covered by focused checks.
- Configured hooks and dependency compatibility checks passed.

Immutable checkpoint revisions, SHA-256 digests, image identity, preprocessing,
and measured output differences are recorded in
[the baseline metadata](validation/onnx-baseline.json). Both real models were
loaded from public safetensors checkpoints at those revisions. The input was
`utils/tests/data/HE.png`, converted to RGB, resized with Pillow bilinear sampling
to 256 × 256, and normalized with the existing per-image min-max helper.

CellPose's original integration assertions pass, with 96 reconstructed instances
and pixel-identical instance/type masks. StarDist reconstructs 79 instances with
pixel-identical instance/type masks, but its original dense assertion fails:
9 of 458,752 values exceed `atol=1e-5, rtol=1e-4`, with maximum absolute error
`4.72e-5`. A follow-up diagnostic using two runtime threads measured 7 outlying
type logits; disabling graph optimization measured 11. This is a remaining
numerical validation gap, not evidence of an incorrect final segmentation on
this image. No tolerance has been loosened.

Main now has an explicit hosted ONNX job that installs compatible
optional packages, tests PyTorch 2.7.1 and 2.14.1, and fails if either runtime
comparison skips. The same gate passed locally with 32 tests and zero skips.
[Linux validation run 37955000998](https://github.com/okunator/cellseg_models.pytorch/actions/runs/37955000998)
was dispatched manually; it validates without publishing. All nine jobs passed,
including both Linux ONNX jobs, both source suites, and all four clean
wheel/source installs. Real-checkpoint comparisons remain local CPU checks. Repeat checkpoint comparisons
on Linux and a representative image set, investigate StarDist's dense differences,
and verify supported accelerator/precision paths separately. The half/double
checks verify sample construction, not end-to-end runtime support for those dtypes.
The single-image CPU record is insufficient to approve runtime dependency upgrades.

## Dependency advisory snapshot

The refreshed 2026-10-09 GitHub snapshot reports 49 open alerts: 27 high,
19 medium, and 3 low, affecting 16 normalized package names in `poetry.lock`.
Pillow appears with two capitalizations in the API; count it once. Severity alone does not establish exploitability. The initial
source-path audit gives the following upgrade groups; no alerts were dismissed.
The table records the initial advisory triage, not a fresh verification of every
patched version. Recheck advisory ranges before selecting an upgrade, including
the newly reported fsspec/download boundary.

| Group | Initial scope and priority |
| --- | --- |
| Image and I/O boundaries | Pillow has 18 alerts, with the newest fixes in 12.3.0. Image conversion, transforms, and downstream decoders make it a priority; confirm the precise affected formats/operations. PyArrow's IPC pre-buffering advisory is fixed in 23.0.1; the repository reads GeoParquet, so inspect the actual backend path rather than equating Parquet with IPC. Its current `^16.1.0` bound prevents that upgrade. |
| Download/checkpoint infrastructure | Upgrade Requests, urllib3, IDNA, and filelock within their compatible ranges and verify downloader behavior and immutable checkpoint loading. Network downloads are a reachable boundary, although each advisory's redirect, compression, temporary-file, or lock preconditions still need checking. |
| PyTorch and model stack | Four alerts affect the locked PyTorch. The source uses `torch.jit.script` in activation helpers; inspect the compilation advisory and the other affected APIs individually. Validate a supported newer PyTorch/timm pair against the prediction baseline before changing runtime bounds. |
| Geospatial packages | GeoPandas 1.1.2 fixes its `to_postgis()` SQL-injection advisory. No direct `to_postgis()` call was found; retain this distinction while upgrading the geospatial stack and testing serialization. |
| Development and build packages | Soup Sieve, pytest, Pygments, virtualenv, fontTools, Black, and setuptools are separate tooling/build work. Check which are reachable through docs/build tooling and upgrade them without bundling model behavior changes. |

This is an initial triage, not a completed vulnerability audit. Record resolved
advisories and compatibility results in each dependency patch. Establish the
representative prediction baseline before the numerical/model upgrades; security
fixes at I/O boundaries can be handled as small independently validated batches.

## Direct dependency and import audit

An AST inventory of library imports, excluding tests and legacy modules, confirms
that Pillow, Hugging Face Hub, safetensors, pandas, Shapely, and NetworkX are used
directly. Merged PR 90 adds them as explicit runtime
requirements while retaining locked versions and artifact hashes. This closes
that metadata gap; optional dataset/training boundaries remain open.

The inventory also found that `from cellseg_models_pytorch.wsi import SlideReader`
failed without development-only Matplotlib. PR 89 is merged and guards both
Matplotlib imports, preserves the visualization dependency error, and adds a
regression plus WSI imports to clean distribution checks. The next smoke check
also imports `inference.predictor`, rather than relying on the empty inference
package initializer. More concrete data/slide APIs still need coverage.

Training/data APIs import Albumentations and PyTables; slide backends use BioIO,
cuCIM, or OpenSlide; optional attention uses xFormers; an alternate StarDist
postprocessor imports the external StarDist package. Inspect each import guard
and caller before declaring extras or making modules lazy. In particular,
`torch_datasets/__init__.py` eagerly imports the training datasets, so importing
`WSIDatasetInfer` also reaches training dependencies. These are separate packaging
fixes, not ONNX regressions, and are not resolved by the merged ONNX corrections.


## Dataset import follow-up

The inference import failure is caused by eager training dataset imports in
`torch_datasets/__init__.py`, an Albumentations-specific annotation in the WSI
dataset, and a constructor dependency check even when no transform is supplied.
The follow-up keeps ordinary exports, postpones annotation-only imports, and
checks optional
dependencies when training datasets are constructed,
and accepts the existing callable transform contract without importing
Albumentations. `WsiSegmenter` now passes `transforms`, matching the dataset API.

Import and sampling coverage is distinct from full slide execution. The existing
`segment()` method references `self.inferer`, which its constructor never defines,
and uses `dst`/`maptype` keywords absent from `BaseModelInst.post_process`.
These confirmed execution gaps remain for a separate inference regression/fix.


## Download and cache security update 2026 10 10

Only four locked packages change. Explicit security minimums prevent consumer
installs from retaining older affected versions. The refreshed snapshot identifies
ten alerts across these packages. Version remediation does not establish that all
exploit preconditions apply to this repository.

| Package | Locked change | Minimum | Advisory IDs | Applicability |
| --- | --- | --- | --- | --- |
| requests | 2.32.4 → 2.34.2 | 2.33.0 | [GHSA-gc5v-m9x4-r6x2](https://github.com/advisories/GHSA-gc5v-m9x4-r6x2) | Zipped CA-bundle extraction is an indirect Requests path; archive/temp-file preconditions were not demonstrated here. |
| urllib3 | 2.5.0 → 2.8.0 | 2.8.0 | [GHSA-vxq7-64xx-v4gw](https://github.com/advisories/GHSA-vxq7-64xx-v4gw), [GHSA-8988-9cw3-xx77](https://github.com/advisories/GHSA-8988-9cw3-xx77), [GHSA-qccp-gfcp-xxvc](https://github.com/advisories/GHSA-qccp-gfcp-xxvc), [GHSA-38jv-5279-wg99](https://github.com/advisories/GHSA-38jv-5279-wg99), [GHSA-2xpw-w6gg-jr37](https://github.com/advisories/GHSA-2xpw-w6gg-jr37), [GHSA-gm62-xv2j-4w53](https://github.com/advisories/GHSA-gm62-xv2j-4w53) | Streaming/decompression are used by Downloader; proxy-specific issues require the corresponding proxy configuration. |
| idna | 3.10 → 3.20 | 3.15 | [GHSA-65pc-fj4g-8rjx](https://github.com/advisories/GHSA-65pc-fj4g-8rjx) | HTTP hostname encoding is reachable; crafted attacker-controlled names are a precondition. |
| filelock | 3.19.1 → 3.32.7 | 3.20.3 | [GHSA-qmgc-5h2g-mvrw](https://github.com/advisories/GHSA-qmgc-5h2g-mvrw), [GHSA-w853-jp5j-5j7f](https://github.com/advisories/GHSA-w853-jp5j-5j7f) | Checkpoint cache locking is reachable; symlink races depend on filesystem/attacker permissions and the selected lock implementation. |

Compatibility checks cover chunked HTTP streaming, gzip decoding, and redirects
against a local server; lock exclusion/release; Unicode hostname encoding; and
both recorded checkpoint cache SHA-256 digests. A 4,973-byte public Hugging Face
README was fetched over HTTPS at the recorded immutable CellPose revision and
read back from cache. No model weights were downloaded in that network smoke.
The upgraded locked offline suite passed 2,047 tests with 5,191 expected skips.

These checks do not reproduce malicious proxy, compression-bomb, or symlink-race
attacks and do not replace broader prediction validation. GitHub can close alerts
after merge and dependency analysis; no alerts were manually dismissed.
