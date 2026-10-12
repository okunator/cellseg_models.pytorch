"""Temporary CPU investigation; remove after downloading the diagnostic outputs."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.models.stardist import StarDist
from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize


def same_partition(a, b):
    if a.shape != b.shape or not np.array_equal(a == 0, b == 0):
        return False
    pairs = np.unique(np.stack((a.ravel(), b.ravel()), axis=1), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


assert same_partition(np.array([0, 1, 1, 2]), np.array([0, 9, 9, 4]))
assert not same_partition(np.array([0, 1, 1, 2]), np.array([0, 9, 4, 4]))
assert not same_partition(np.array([0, 1, 1, 2]), np.array([9, 0, 0, 4]))

root = Path(__file__).resolve().parents[1]
data = root / "tests/fixtures/checkpoint_baseline"
meta = json.loads((data / "metadata.json").read_text())
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2)
with Image.open(root / meta["image"]) as source:
    image = source.convert("RGB").resize(tuple(meta["size"]), Image.Resampling.BILINEAR)
x = (
    torch.from_numpy(minmax_normalize(np.asarray(image), copy=True).transpose(2, 0, 1))
    .unsqueeze(0)
    .contiguous()
)
report = {"platform": platform.platform(), "torch": torch.__version__, "models": {}}
for family, cls in [("cellpose", CellPose), ("stardist", StarDist)]:
    spec = meta["models"][family]["checkpoint"]
    weights = hf_hub_download(
        repo_id=spec["repo_id"], filename=spec["filename"], revision=spec["revision"]
    )
    assert hashlib.sha256(Path(weights).read_bytes()).hexdigest() == spec["sha256"]
    model = cls.from_pretrained(weights, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)
    if family == "cellpose":
        model.post_processor = PostProcessor("cellpose", {"use_gpu": False})
    report["models"][family] = {}
    for backend, enabled, dtype in [
        ("default", True, torch.float32),
        ("native", False, torch.float32),
        ("float64", False, torch.float64),
    ]:
        model.model.to(dtype=dtype)
        with torch.backends.mkldnn.flags(enabled=enabled), torch.inference_mode():
            soft = model.model(x.to(dtype=dtype))["nuc"]
            dense = (
                (soft.aux_map, soft.type_map)
                if family == "cellpose"
                else (soft.binary_map, soft.aux_map, soft.type_map)
            )
            inst, types = model.post_process(model.predict(x.to(dtype=dtype)))["nuc"][0]
        actual = {f"dense_{i}": tensor.numpy() for i, tensor in enumerate(dense)}
        actual.update(instance_mask=inst, type_mask=types)
        np.savez_compressed(out / f"{family}-{backend}.npz", **actual)
        with np.load(data / f"{family}.npz", allow_pickle=False) as reference:
            result = {
                "instance_label_changes": int(
                    np.count_nonzero(inst != reference["instance_mask"])
                ),
                "same_partition": bool(
                    same_partition(inst, reference["instance_mask"])
                ),
                "class_changes": int(np.count_nonzero(types != reference["type_mask"])),
                "instances": int(len(np.unique(inst)) - 1),
                "dense": [],
            }
            for i in range(len(dense)):
                a, b = actual[f"dense_{i}"], reference[f"dense_{i}"]
                error = np.abs(a - b)
                result["dense"].append(
                    {
                        "max_abs": float(error.max()),
                        "outside_bounds": int(
                            np.count_nonzero(
                                error
                                > meta["dense_atol"] + meta["dense_rtol"] * np.abs(b)
                            )
                        ),
                    }
                )
        report["models"][family][backend] = result
        print(family, backend, json.dumps(result), flush=True)
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
