"""Replay pinned dense predictions to measure the existing CPU/CUDA boundary."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import numpy as np
import torch
from numba import get_num_threads

from cellseg_models_pytorch.decoders.multitask_decoder import SoftInstanceOutput
from cellseg_models_pytorch.inference.post_processor import PostProcessor
from cellseg_models_pytorch.utils import majority_vote_sequential

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/checkpoint_baseline"
T = TypeVar("T")


def load_case(family: str, device: torch.device) -> tuple[SoftInstanceOutput, dict]:
    """Load verified BCHW float32 heads and HW int32 reference labels offline."""
    metadata = json.loads((FIXTURES / "metadata.json").read_text())
    record = metadata["models"][family]
    path = FIXTURES / f"{family}.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["reference_sha256"]:
        raise ValueError(f"Reference checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    offset = int(family == "stardist")
    classes = torch.from_numpy(arrays[f"dense_{offset + 1}"]).softmax(1).argmax(1)
    output = SoftInstanceOutput(
        type_map=classes.to(device),
        aux_map=torch.from_numpy(arrays[f"dense_{offset}"]).to(device),
        binary_map=(torch.from_numpy(arrays["dense_0"]).to(device) if offset else None),
    )
    return output, {
        "instances": arrays["instance_mask"],
        "classes": arrays["type_mask"],
        "sha256": digest,
        "checkpoint": record["checkpoint"],
    }


def check_result(result: tuple[np.ndarray, np.ndarray], reference: dict) -> None:
    """Reject changed background, object partitions, classes or output dtypes."""
    instances, classes = result
    expected = reference["instances"]
    if instances.dtype != expected.dtype or instances.shape != expected.shape:
        raise ValueError("Instance shape or dtype differs from the reference")
    if np.any(instances < 0) or not np.array_equal(instances == 0, expected == 0):
        raise ValueError("Instance background differs from the reference")
    pairs = np.unique(np.stack((instances.ravel(), expected.ravel()), axis=1), axis=0)
    if not (len(pairs) == len(np.unique(instances)) == len(np.unique(expected))):
        raise ValueError("Instance partition differs from the reference")
    if classes.dtype != reference["classes"].dtype or not np.array_equal(
        classes, reference["classes"]
    ):
        raise ValueError("Class labels or dtype differ from the reference")


def timed(func: Callable[[], T], device: torch.device) -> tuple[T, float]:
    """Measure completed work, including device transfers, in wall milliseconds."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = func()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, (perf_counter() - start) * 1000


def profile_case(
    family: str, device: torch.device, warmup: int = 2, repeats: int = 5
) -> dict:
    """Time one pinned tile, retaining only per-stage scalar samples."""
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU and CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; no CPU fallback")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context, torch.inference_mode():
        output, reference = load_case(family, device)
        processor = PostProcessor(
            family, {"use_gpu": device.type == "cuda"} if family == "cellpose" else {}
        )
        maps = {"nuc": output, "cyto": None, "tissue": None}
        samples = {name: [] for name in ("prepare", "reconstruct", "classify", "total")}
        for iteration in range(warmup + repeats):
            if iteration == warmup and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            prepared, prepare_ms = timed(
                lambda: processor._prepare_inst_maps(output), device
            )
            foreground, auxiliary, types = (values[0] for values in prepared)
            instances, reconstruct_ms = timed(
                lambda: processor.postproc_func(foreground, auxiliary).astype("i4"),
                device,
            )
            classes, classify_ms = timed(
                lambda: majority_vote_sequential(types, instances).astype("i4"), device
            )
            check_result((instances, classes), reference)
            full, total_ms = timed(lambda: processor.postproc_serial(maps), device)
            check_result(full["nuc"][0], reference)
            if iteration >= warmup:
                for name, elapsed in zip(
                    samples, (prepare_ms, reconstruct_ms, classify_ms, total_ms)
                ):
                    samples[name].append(elapsed)
        # CPU input conversion copies class maps twice without a binary head.
        inputs = (
            output.binary_map if output.binary_map is not None else output.type_map,
            output.aux_map,
            output.type_map,
        )
        report = {
            "family": family,
            "device": str(device),
            "shape_bchw": list(output.aux_map.shape),
            "input_dtype": str(output.aux_map.dtype),
            "integration_dtype": "float64" if family == "cellpose" else None,
            "warmup": warmup,
            "repeats": repeats,
            "timings_ms": {
                name: {"samples": values, "median": statistics.median(values)}
                for name, values in samples.items()
            },
            "prepare_device_to_host_bytes": (
                sum(t.numel() * t.element_size() for t in inputs)
                if device.type == "cuda"
                else 0
            ),
            "prepared_array_bytes": sum(
                a.nbytes for values in prepared for a in values
            ),
            "foreground_pixels": int(np.count_nonzero(reference["instances"])),
            "instances": int(len(np.unique(reference["instances"])) - 1),
            "reference_sha256": reference["sha256"],
            "checkpoint": reference["checkpoint"],
            "parity": "exact partitions, background and classes; positive IDs may permute",
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("cellpose", "stardist"), required=True)
    parser.add_argument(
        "--device", default="cpu", help="cpu or cuda:N; never auto-selected"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON report path"
    )
    args = parser.parse_args()
    if os.environ.get("NUMBA_DISABLE_JIT", "0") != "0":
        parser.error("Unset NUMBA_DISABLE_JIT for performance measurements")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    report = profile_case(args.family, device, args.warmup, args.repeats)
    report["environment"] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
            ).strip()
        ),
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "numpy",
                "scipy",
                "numba",
                "scikit-image",
                "cellseg-models-pytorch",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "numba_threads": get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if sys.platform != "win32":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = (
            rss if sys.platform == "darwin" else rss * 1024
        )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Verified {args.family} reconstruction; wrote {args.output}")


if __name__ == "__main__":
    main()
