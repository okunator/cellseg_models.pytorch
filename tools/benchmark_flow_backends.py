"""Compare original and tensor flow reconstruction on pinned Hub dataset crops."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from numba import get_num_threads
from PIL import Image

from cellseg_models_pytorch.metrics import panoptic_quality
from cellseg_models_pytorch.models.cellpose import CellPose
from cellseg_models_pytorch.postproc import (
    fill_holes_and_remove_small_masks,
    follow_flows_torch,
    get_masks_cellpose,
    get_masks_omnipose,
)
from cellseg_models_pytorch.postproc.functional.cellpose.integrator import follow_flows
from cellseg_models_pytorch.postproc.functional.omnipose import div_rescale
from cellseg_models_pytorch.transforms import minmax_normalize
from cellseg_models_pytorch.transforms.functional import (
    gen_flow_maps,
    gen_omni_flow_maps,
)
from cellseg_models_pytorch.utils import binarize, majority_vote_sequential

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    {
        "name": "pannuke",
        "folder": "pannuke-parquet",
        "file": "fold_3.parquet",
        "revision": "8e8bf584dfd8bacc5255e22cd5d32c14094ed0df",
        "sha256": "98327000b6203ac08c7a80288024653049ab764b203b1dd133fee9c467440001",
    },
    {
        "name": "panoptils_refined",
        "folder": "panoptils-refined-splits",
        "file": "test.parquet",
        "revision": "21cc3a3985117c84fb399f0048920eba42fbba32",
        "sha256": "c77b1fca5d2cc0fa879a61062573dc27d7839c9fb9617a9173b242291b04bed3",
    },
    {
        "name": "cytonuke",
        "folder": "cytonuke-parquet",
        "file": "test.parquet",
        "revision": "c18da9e0f7bc0b75610af0c45f04b4ead1763f11",
        "sha256": "c4fd8a00f6cd47348d2f3e27956b5aa446c59dcafa4a1604a5ab59bad3cfa361",
    },
]
CHECKPOINT = {
    "repo_id": "csmp-hub/cellpose-histo-hgsc-nuc-v1",
    "revision": "88c7ffd3a9106d1673c3dd6e6a9e42476e8320d2",
    "filename": "cellpose_hgsc_v1_efficientnet_b5.safetensors",
    "sha256": "d6e29abb0b1c60668961f1bf074e609091583cae8340f1360e6ccfcb1cf17bb4",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_partition(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Compare object pixels/background while allowing positive IDs to permute."""
    if actual.shape != expected.shape or np.any(actual < 0):
        return False
    if not np.array_equal(actual == 0, expected == 0):
        return False
    pairs = np.unique(np.stack((actual.ravel(), expected.ravel()), axis=1), axis=0)
    return len(pairs) == len(np.unique(actual)) == len(np.unique(expected))


def measure_pair(
    functions: dict[str, Callable], warmup: int, repeats: int
) -> dict[str, dict]:
    """Alternate method order and retain completed CPU wall-time samples."""
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    samples = {name: [] for name in functions}
    names = list(functions)
    for iteration in range(warmup + repeats):
        for name in names[:: (-1 if iteration % 2 else 1)]:
            start = perf_counter_ns()
            functions[name]()
            elapsed = (perf_counter_ns() - start) / 1e6
            if iteration >= warmup:
                samples[name].append(elapsed)
    return {
        name: {"samples_ms": values, "median_ms": statistics.median(values)}
        for name, values in samples.items()
    }


def reconstruct(
    family: str,
    endpoints: np.ndarray,
    indices: np.ndarray,
    mask: np.ndarray,
    types: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "cellpose":
        instances = get_masks_cellpose(endpoints)
    else:
        instances = get_masks_omnipose(endpoints, mask, indices)
    instances = fill_holes_and_remove_small_masks(instances, min_size).astype(np.int32)
    classes = majority_vote_sequential(types, instances).astype(np.int32)
    return instances, classes


def compare_case(
    family: str,
    field: np.ndarray,
    mask: np.ndarray,
    types: np.ndarray,
    truth: np.ndarray,
    min_size: int,
    warmup: int,
    repeats: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Compare identical prepared fields and common CPU grouping/cleanup."""
    tensor = torch.from_numpy(field)
    tensor_mask = torch.from_numpy(mask)
    suppress = family == "omnipose"

    def legacy():
        return follow_flows(
            field, mask, niter=200, suppress_euler=suppress, use_gpu=False
        )

    def native():
        return follow_flows_torch(
            tensor, tensor_mask, niter=200, suppress_euler=suppress
        )

    old_points, old_indices = legacy()
    points, indices = native()
    new_points, new_indices = points.numpy(), indices.numpy()
    old_masks = reconstruct(family, old_points, old_indices, mask, types, min_size)
    new_masks = reconstruct(family, new_points, new_indices, mask, types, min_size)
    core = measure_pair({"original": legacy, "tensor": native}, warmup, repeats)

    def pipeline(method):
        p, inds = method()
        if isinstance(p, torch.Tensor):
            p, inds = p.numpy(), inds.numpy()
        return reconstruct(family, p, inds, mask, types, min_size)

    total = measure_pair(
        {"original": lambda: pipeline(legacy), "tensor": lambda: pipeline(native)},
        warmup,
        repeats,
    )
    summary = {
        "family": family,
        "shape_hw": list(mask.shape),
        "flow_dtype": str(field.dtype),
        "min_size": min_size,
        "seed_pixels": len(old_indices),
        "truth_objects": int(np.count_nonzero(np.unique(truth))),
        "foreground_fraction": float(mask.mean()),
        "endpoint_max_abs_diff": float(np.max(np.abs(old_points - new_points))),
        "seed_indices_equal": bool(np.array_equal(old_indices, new_indices)),
        "partition_equal": same_partition(old_masks[0], new_masks[0]),
        "classes_equal": bool(np.array_equal(old_masks[1], new_masks[1])),
        "outputs": {
            name: {
                "objects": int(np.count_nonzero(np.unique(pair[0]))),
                **panoptic_quality(truth, pair[0]),
            }
            for name, pair in (("original", old_masks), ("tensor", new_masks))
        },
        "integration": core,
        "reconstruction_from_prepared_flows": total,
    }
    arrays = {
        "field": field,
        "seed_mask": mask,
        "types": types,
        "truth": truth,
        "original_endpoints": old_points,
        "tensor_endpoints": new_points,
        "original_instances": old_masks[0],
        "tensor_instances": new_masks[0],
        "original_classes": old_masks[1],
        "tensor_classes": new_masks[1],
    }
    return summary, arrays


def read_row(file: pq.ParquetFile, index: int) -> dict:
    if index < 0:
        raise IndexError("Row index must be nonnegative")
    for group in range(file.metadata.num_row_groups):
        count = file.metadata.row_group(group).num_rows
        if index < count:
            return (
                file.read_row_group(group, columns=["id", "image", "inst", "type"])
                .slice(index, 1)
                .to_pylist()[0]
            )
        index -= count
    raise IndexError("Row index outside the dataset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New results directory"
    )
    parser.add_argument(
        "--samples", type=int, default=6, help="Evenly spaced source rows per dataset"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.repeats) < 1:
        parser.error("Sample, warmup and repetition counts must be positive")
    if os.environ.get("NUMBA_DISABLE_JIT", "0") != "0":
        parser.error("Numba JIT must be enabled for performance comparisons")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    checkpoint = Path(
        hf_hub_download(
            **{k: CHECKPOINT[k] for k in ("repo_id", "revision", "filename")}
        )
    )
    if sha256(checkpoint) != CHECKPOINT["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model = CellPose.from_pretrained(checkpoint, device=torch.device("cpu"))
    model.set_inference_mode(mixed_precision=False)
    report = {
        "checkpoint": CHECKPOINT,
        "datasets": DATASETS,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if platform.system() == "Darwin"
            else platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "numba_threads": get_num_threads(),
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "numba",
                    "scipy",
                    "scikit-learn",
                    "cellseg-metrics",
                )
            },
            "source_commit": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_dirty": bool(
                subprocess.check_output(
                    ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
                ).strip()
            ),
            "script_sha256": sha256(Path(__file__)),
            "implementation_sha256": {
                "original": sha256(
                    ROOT
                    / "cellseg_models_pytorch/postproc/functional/cellpose/integrator.py"
                ),
                "tensor": sha256(
                    ROOT / "cellseg_models_pytorch/postproc/torch/flow.py"
                ),
            },
            "load_average_start": os.getloadavg(),
            "power_source": subprocess.check_output(
                ["pmset", "-g", "batt"], text=True
            ).splitlines()[0]
            if platform.system() == "Darwin"
            else None,
        },
        "protocol": {
            "device": "cpu",
            "batch_size": 1,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "iterations": 200,
            "samples_per_dataset": args.samples,
            "crop_sizes": [256, 512],
            "crop": "native center crop, no resizing; cut objects retain IDs",
            "order": "alternating backend order per repetition",
            "model_precision": "float32, no AMP",
            "integration_precision": "float64, float32 endpoints",
            "quality": "per-crop PQ at strict IoU > 0.5, no class taxonomy comparison",
            "omnipose_predictions": "not tested: no compatible trained Omnipose checkpoint available",
            "timing_exclusions": "model loading/inference, target generation, map preparation, input conversion, metrics and file I/O",
        },
        "cases": [],
    }
    with (args.output / "cases.jsonl").open("x") as log:
        for dataset in DATASETS:
            path = args.data_root / dataset["folder"] / dataset["file"]
            if sha256(path) != dataset["sha256"]:
                raise ValueError(f"Dataset checksum mismatch: {path}")
            file = pq.ParquetFile(path)
            rows = np.unique(
                np.linspace(0, file.metadata.num_rows - 1, args.samples, dtype=int)
            )
            for row_index in rows:
                row = read_row(file, int(row_index))
                with Image.open(BytesIO(row["image"])) as source:
                    image = np.array(source)
                with Image.open(BytesIO(row["inst"])) as source:
                    truth = np.array(source).astype(np.int32)
                with Image.open(BytesIO(row["type"])) as source:
                    target_types = np.array(source).astype(np.int32)
                if (
                    image.ndim != 3
                    or image.shape[2] != 3
                    or image.shape[:2] != truth.shape
                    or truth.shape != target_types.shape
                ):
                    raise ValueError("Expected aligned RGB image and integer HW masks")
                for size in (256, 512):
                    if min(truth.shape) < size:
                        continue
                    y, x = ((s - size) // 2 for s in truth.shape)
                    crop = image[y : y + size, x : x + size].copy()
                    gt = truth[y : y + size, x : x + size].copy()
                    gt_types = target_types[y : y + size, x : x + size].copy()
                    tensor = (
                        torch.from_numpy(
                            minmax_normalize(crop, copy=True).transpose(2, 0, 1)
                        )
                        .unsqueeze(0)
                        .contiguous()
                    )
                    with torch.inference_mode():
                        prediction = model.predict(tensor)["nuc"]
                    pred_types = prediction.type_map[0].numpy().astype(np.int32)
                    target_mask = gt > 0
                    pred_mask = binarize(pred_types).astype(bool)
                    fields = [
                        (
                            "cellpose",
                            "target",
                            gen_flow_maps(gt),
                            target_mask,
                            gt_types,
                            0,
                        ),
                        (
                            "cellpose",
                            "prediction",
                            prediction.aux_map[0].numpy() * pred_mask,
                            pred_mask,
                            pred_types,
                            30,
                        ),
                        (
                            "omnipose",
                            "target",
                            div_rescale(gen_omni_flow_maps(gt), target_mask),
                            target_mask,
                            gt_types,
                            0,
                        ),
                    ]
                    for family, source, field, mask, types, min_size in fields:
                        result, arrays = compare_case(
                            family,
                            field,
                            mask,
                            types,
                            gt,
                            min_size,
                            args.warmup,
                            args.repeats,
                        )
                        case = f"{dataset['name']}-{int(row_index):04d}-{size}-{family}-{source}"
                        result.update(
                            {
                                "case": case,
                                "dataset": dataset["name"],
                                "sample_id": row["id"],
                                "row_index": int(row_index),
                                "crop_yxhw": [y, x, size, size],
                                "source": source,
                            }
                        )
                        np.savez_compressed(args.output / f"{case}.npz", **arrays)
                        result["artifact_sha256"] = sha256(args.output / f"{case}.npz")
                        log.write(json.dumps(result) + "\n")
                        log.flush()
                        report["cases"].append(result)
                        print(
                            f"{case}: partitions={result['partition_equal']}, endpoint diff={result['endpoint_max_abs_diff']}",
                            flush=True,
                        )
    report["environment"]["load_average_end"] = os.getloadavg()
    if platform.system() != "Windows":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["environment"]["process_peak_rss_bytes"] = (
            rss if platform.system() == "Darwin" else rss * 1024
        )
    report["all_parity_passed"] = all(
        case["partition_equal"]
        and case["classes_equal"]
        and case["seed_indices_equal"]
        and case["endpoint_max_abs_diff"] == 0
        for case in report["cases"]
    )
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    if not report["all_parity_passed"]:
        raise SystemExit("Comparison differences recorded; investigate report.json")


if __name__ == "__main__":
    main()
