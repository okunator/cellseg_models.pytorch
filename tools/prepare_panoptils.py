"""Prepare patient-separated nucleus patches from the pinned PanopTILs archive."""

import argparse
import hashlib
import json
import random
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy.io import savemat

DATASET = "histolytics-hub/panoptils_refined"
REVISION = "d40293bc1ac0d0da8f271959fdb990aeea2d884f"
PARQUET_SHA256 = "1ab3f0c2299e0c44f996e34c233846592a934aa3dcfc8e80cc983db85c771d18"
PATCH_SIZE = 256
SEED = 42


def _patient(slide: str) -> str:
    parts = slide.split("-")
    if len(parts) < 4 or parts[0] != "TCGA":
        raise ValueError(f"Unrecognized TCGA slide identifier: {slide!r}")
    return "-".join(parts[:3])


def _decode_png(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.array(image)


def prepare_panoptils(source: Path, destination: Path) -> None:
    """Write aligned 256-pixel patches after a seeded patient-level split.

    Args:
        source: Pinned parquet archive with raw PNG image/inst/type fields.
        destination: New folder dataset; existing directories are never overwritten.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the archive, identifiers, images or nucleus labels are invalid.
    """
    if destination.exists():
        raise FileExistsError(destination)
    digest = hashlib.sha256()
    with source.open("rb") as archive:
        for block in iter(lambda: archive.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != PARQUET_SHA256:
        raise ValueError("Archive identity changed; review before changing its digest")
    parquet = pq.ParquetFile(source)
    slides = parquet.read(columns=["slide_name"]).column(0).to_pylist()
    patients = sorted({_patient(slide) for slide in slides})
    if len(patients) < 2:
        raise ValueError("Need at least two patients for separate splits")
    random.Random(SEED).shuffle(patients)
    validation_patients = set(patients[: max(1, len(patients) // 5)])
    counts = {
        split: {"regions": 0, "patches": 0, "class_pixels": [0] * 7}
        for split in ("train", "validation")
    }
    records = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".panoptils-") as temporary:
        root = Path(temporary)
        for split in counts:
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir()
        (root / "validation/regions/images").mkdir(parents=True)
        (root / "validation/regions/labels").mkdir()
        batches = parquet.iter_batches(batch_size=1, columns=["image", "inst", "type"])
        for row_index, batch in enumerate(batches):
            row = batch.to_pylist()[0]
            arrays = {key: _decode_png(row[key]) for key in ("image", "inst", "type")}
            image, instances, classes = arrays["image"], arrays["inst"], arrays["type"]
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Row {row_index}: expected RGB uint8 pixels")
            if instances.shape != image.shape[:2] or classes.shape != instances.shape:
                raise ValueError(f"Row {row_index}: image and masks are not aligned")
            if instances.dtype.kind not in "ui" or classes.dtype.kind not in "ui":
                raise ValueError(f"Row {row_index}: labels must be integers")
            if (
                instances.min() < 0
                or instances.max() > np.iinfo(np.int32).max
                or classes.min() < 0
                or classes.max() > 6
            ):
                raise ValueError(f"Row {row_index}: invalid instance or class IDs")
            if classes[instances == 0].any():
                raise ValueError(
                    f"Row {row_index}: background has nonzero cell classes"
                )
            height, width = instances.shape
            if (
                min(height, width) < PATCH_SIZE
                or height % PATCH_SIZE
                or width % PATCH_SIZE
            ):
                raise ValueError(
                    f"Row {row_index}: dimensions must be multiples of {PATCH_SIZE}"
                )
            patient = _patient(slides[row_index])
            split = "validation" if patient in validation_patients else "train"
            region_name = f"region_{row_index:04d}"
            if split == "validation":
                Image.fromarray(image).save(
                    root / "validation/regions/images" / f"{region_name}.png"
                )
                savemat(
                    root / "validation/regions/labels" / f"{region_name}.mat",
                    {"inst": instances.astype(np.int32), "type": classes},
                    do_compression=True,
                )
            counts[split]["regions"] += 1
            pixels = np.bincount(classes.ravel(), minlength=7)
            counts[split]["class_pixels"] = (
                np.array(counts[split]["class_pixels"]) + pixels
            ).tolist()
            for y in range(0, height, PATCH_SIZE):
                for x in range(0, width, PATCH_SIZE):
                    area = np.s_[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    name = f"{region_name}_y{y:04d}_x{x:04d}"
                    Image.fromarray(image[area]).save(
                        root / split / "images" / f"{name}.png"
                    )
                    savemat(
                        root / split / "labels" / f"{name}.mat",
                        {
                            "inst": instances[area].astype(np.int32),
                            "type": classes[area],
                        },
                        do_compression=True,
                    )
                    counts[split]["patches"] += 1
            records.append(
                {
                    "row": row_index,
                    "patient": patient,
                    "split": split,
                    "file": region_name,
                }
            )
            if (row_index + 1) % 100 == 0:
                print(f"Prepared {row_index + 1} regions", flush=True)
        manifest = {
            "dataset": DATASET,
            "revision": REVISION,
            "sha256": PARQUET_SHA256,
            "patch_size": PATCH_SIZE,
            "seed": SEED,
            "samples": records,
            "splits": counts,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        root.rename(destination)
    for split, count in counts.items():
        print(f"{split}: {count['regions']} regions, {count['patches']} patches")


def main() -> None:
    """Prepare a new folder dataset from a previously downloaded archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_panoptils(args.source, args.destination)


if __name__ == "__main__":
    main()
