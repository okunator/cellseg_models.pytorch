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
class IDs are preserved as integers 0–6, with background 0. The fixture does not
assign unverified semantic names to those IDs.

Each NPZ contains dense float32 BCHW outputs, an int32 HW instance-label mask,
and an int32 HW class mask. CellPose outputs are two flow channels and class logits;
StarDist outputs are raw foreground scores, 32 radial-distance channels, and
class logits. Foreground scores are unconstrained head outputs, not calibrated
probabilities. Masks preserve instance IDs and background 0. Expected instance counts
are 96 and 79, respectively. Dense comparison uses `atol=1e-5, rtol=1e-4`;
instance and class masks must match exactly. These bounds do not replace or
relax the separate ONNX parity tests.

Run the offline fixture integrity check with:

```sh
HF_HUB_OFFLINE=1 NUMBA_DISABLE_JIT=1 poetry run pytest cellseg_models_pytorch/models/tests/test_checkpoint_baseline.py
```

Run the explicit checkpoint integration check with:

```sh
NUMBA_DISABLE_JIT=1 poetry run pytest --optional cellseg_models_pytorch/models/tests/test_checkpoint_baseline.py
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
a quality benchmark, or GPU/AMP validation. Registered model-name loading,
additional images/model families, and the StarDist ONNX tolerance gap remain
separate work.

The first Linux x86-64 run of this preserved baseline passed CellPose but found
two StarDist class logits outside the dense bounds (maximum absolute difference
2.193451e-5). Keep this visible while investigating numerical backend differences;
the gate is not ready for numerical upgrades. Subtests report every dense map,
mask, and count even when another comparison fails. No reference or tolerance
has been changed in response to this failure.
