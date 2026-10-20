# CPU checkpoint baseline

These fixtures preserve the CellPose and StarDist PyTorch arrays recorded on
2026-10-09 on macOS arm64 / Python 3.11, before the maintenance dependency
upgrades. They are compressed
copies of those arrays, not predictions regenerated to make an upgrade pass.
`metadata.json` pins the public Hugging Face checkpoint revisions and SHA-256
checksums, the repository image, preprocessing, reference files, and model settings.

The input is the existing `utils/tests/data/HE.png`, converted to RGB, resized
with PIL bilinear interpolation to 256 × 256, and min-max normalized per image.
Inference uses a CPU float32 BCHW tensor, evaluation mode, no AMP, and two
PyTorch threads. Both checkpoints use EfficientNet-B5 and seven class channels;
class IDs are preserved as integers 0–6. Background pixels have class 0;
a reconstructed instance can also receive class 0. The fixture does not
assign unverified semantic names to those IDs.

Each NPZ contains dense float32 BCHW outputs, an int32 HW instance-label mask,
and an int32 HW class mask. CellPose outputs are two flow channels and class logits;
StarDist outputs are raw foreground scores, 32 radial-distance channels, and
class logits. Foreground scores are unconstrained head outputs, not calibrated
probabilities. Masks preserve instance IDs and background 0. Expected instance counts
are 96 and 79, respectively. Dense comparison uses `atol=1e-5, rtol=1e-4`;
Flow, ray, and foreground outputs are compared in their raw units. Class-head
probabilities are evaluated along the class axis in float64 with the same bounds.
Class masks and instance dtype, background, object pixels, and counts must match
exactly; a bijection of positive instance IDs is allowed. Predictions and stored
reference IDs are not rewritten. These bounds do not replace or
relax the separate ONNX parity tests.

Run the offline fixture integrity check with:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --no-sync pytest tests/test_checkpoint_baseline.py
```

For numerical upgrades and releases, run the checkpoint comparison manually.
This checks one image and two checkpoints; it does not establish representative
dataset coverage. There is no recurring checkpoint-download CI
job. The offline integrity check remains in ordinary CI. These repo-level test
fixtures are not shipped in wheels or source distributions.

Run the explicit checkpoint integration check with:

```sh
NUMBA_DISABLE_JIT=1 uv run --no-sync pytest --optional tests/test_checkpoint_baseline.py
```

This downloads only the pinned safetensors files into the Hugging Face cache.
Set `CELLSEG_BASELINE_CACHE` to reuse another cache; add `HF_HUB_OFFLINE=1` for
cached-only runs. Missing files and failed downloads fail the integration check.
Ordinary tests do not download checkpoints. The `--optional` selection skips the
ordinary integrity test, so run both commands.

The image remains under the repository's existing attribution and permissions.
The model repositories are
[CellPose](https://huggingface.co/csmp-hub/cellpose-histo-hgsc-nuc-v1) and
[StarDist](https://huggingface.co/csmp-hub/stardist-histo-hgsc-nuc-v1).
No model weights are stored here. Do not overwrite reference arrays to conceal
regressions; investigate differences and review any scientifically justified
baseline change separately.

This is one recorded image and two checkpoints, not a representative dataset,
a quality benchmark, or GPU/AMP validation. Registered model-name loading
and additional images/model families remain separate work. The StarDist ONNX
class-head calibration was resolved in [PR 101](https://github.com/okunator/cellseg_models.pytorch/pull/101);
its separate raw-head tolerances do not replace these native baseline bounds.

The CPU comparison policy follows the inference contract: class logits pass
through softmax before classification, and instance numbers identify objects
within an independently inferred image. A common logit offset leaves class
probabilities unchanged, but changed confidence must still fail. Split, merged,
moved, or missing objects must fail even when counts or classes match. Unit tests
exercise these distinctions. [SciPy's softmax definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html)
describes the class-probability transformation.

A one-off Linux/macOS investigation found identical object partitions and class
maps for default/native float32 and float64 runs. Raw class logits showed backend
rounding differences, while probabilities met the existing bounds. The CPU policy
does not alter production inference, checkpoint/reference files, or the separate
ONNX raw-output assertions. Numerical backend equivalence is limited to the
recorded inputs; GPU/AMP and broader datasets remain unverified.

## Pinned pretrained ONNX checks

[The separate pretrained ONNX checks](../../test_pretrained_onnx.py) pin the HGSC
HoVer-Net and CellVit repository revisions and SHA256 hashes. Only those exact
weights use class-logit absolute tolerances of `2e-5` and `4e-5`, respectively.
Relative tolerance stays `1e-4`; reconstruction maps retain `atol=1e-5`.
Raw logits, raw/public class decisions and exact object pixels/background and
class masks remain checked. Only positive instance-ID permutations are allowed.
Ordinary untrained-export tolerances and the CPU fixtures above are unchanged.

These budgets follow CPU native float32/float64 calibration on eight tissue fields
from TCGA-AC-A6IW and CMU-1, frozen before validation on four separate fields and
two existing image controls. The native-derived two-implementation allowances
were `1.7152e-5` and `3.3489e-5`, rounded up to `1e-5` increments. Both budgets
passed 34 fixed/dynamic ONNX cases with exact classes/masks. Calibration used
PyTorch 2.14.1, timm 1.0.30 and ONNX Runtime 1.31.0 on Linux ARM64; this does not
establish budgets for other checkpoints or GPU/mixed-precision inference.

The maintained integration check uses the existing HE image, OpenCV linear resize
to 256 pixels and global float32 min-max normalization. CellVit then takes the
native center 224 crop. It exports fresh fixed/dynamic graphs and checks batches
one, two and three. Install `tools/onnx-requirements.txt`, then run explicitly:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 uv run --no-sync pytest tests/test_pretrained_onnx.py
NUMBA_DISABLE_JIT=1 uv run --no-sync pytest --optional tests/test_pretrained_onnx.py -k pinned_pretrained_onnx_parity
```

The optional command downloads only the pinned weights unless they are cached.
Reuse `CELLSEG_BASELINE_CACHE` and add `HF_HUB_OFFLINE=1` for cached-only checks.
No recurring checkpoint-download CI job is added; the ordinary regressions stay offline.
