"""Prepare a pinned, patient-separated PanopTILs subset for the training tutorial."""

import argparse
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.io import savemat

DATASET = "histolytics-hub/panoptils_refined"
REVISION = "d40293bc1ac0d0da8f271959fdb990aeea2d884f"
SELECTION = {0: "train", 1: "train", 24: "train", 25: "validation"}
PNG_SHA256 = {
    0: {
        "image": "239966c832e729c48a68f15ea2ff5967d0a82548fcadabd3aad8a051e1ff37eb",
        "inst": "97b7e644553bcad3fae5a492240c28647fc4409f423a597039a1bd63078f6bbf",
        "type": "8ceb53437e46c8dfd7d3f8cf2f410b7ab1dd4665ce8e7a4a4b9af1a29afc7393",
    },
    1: {
        "image": "78a962a5bdf57eea8cf100e2eae729d3b942c2022444d77e320185eaf5eb1718",
        "inst": "c44e17dc8c4af1ab5e41aa0593f87fa9729ef43aaa461266717a907cc310d05d",
        "type": "e560c04a5158615b740fb37551444ec1daaba5173d0ae8423fdd60d9e97b35c9",
    },
    24: {
        "image": "6a518e18c5445315b09df4ffd1b1c484f28beb424bbc0f50e55437d1fe19eb87",
        "inst": "095a21b9dd67b48cff9decad7a6fa6b7cb2e499c66d8be9c25e59ad5f4b9fc58",
        "type": "7775d6b24fd17126243e9b810645dc2bdd891d3c294567da8dfaca41d2ccd8a7",
    },
    25: {
        "image": "a1efe6262ddc63994cf47cebb7182312697af9251ddcc686251a5c5d66445310",
        "inst": "a07144a50875e5540287b20783a7e6ea58b1c94cd50b72e70d39822b0109490d",
        "type": "8be2ffa7d890ca42a41eae260c7782fc2d48f8fac181bd8fd437fd86f20f6e4a",
    },
}


def _decode_png(encoded: str, expected_hash: str) -> np.ndarray:
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(
            "Dataset sample changed; review its identity before updating hashes."
        )
    with Image.open(BytesIO(payload)) as image:
        return np.array(image)


def prepare_panoptils(destination: Path) -> None:
    """Download four regions, crop aligned labels and write a new folder dataset.

    Args:
        destination: New output directory; existing directories are never overwritten.

    Raises:
        ValueError: If sample bytes, labels or patient separation are invalid.
        FileExistsError: If the output directory already exists.

    Example:
        prepare_panoptils(Path("data/panoptils"))
    """
    if destination.exists():
        raise FileExistsError(destination)
    samples = []
    patients = {"train": set(), "validation": set()}
    for offset, split in SELECTION.items():
        response = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": 1,
            },
            timeout=60,
        )
        response.raise_for_status()
        row = response.json()["rows"][0]["row"]
        parts = row["slide_name"].split("-")
        if len(parts) < 4 or parts[0] != "TCGA":
            raise ValueError("Expected a TCGA slide identifier with a patient prefix.")
        patient = "-".join(parts[:3])
        patients[split].add(patient)

        # Tissue-semantic masks are intentionally never decoded or saved.
        arrays = {
            key: _decode_png(row[key], digest)
            for key, digest in PNG_SHA256[offset].items()
        }
        image, instances, classes = arrays["image"], arrays["inst"], arrays["type"]
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or instances.shape != image.shape[:2]
            or classes.shape != instances.shape
            or min(instances.shape) < 256
        ):
            raise ValueError(
                "Expected aligned RGB images and masks at least 256 pixels wide/high."
            )
        if (
            instances.dtype.kind not in "ui"
            or classes.dtype.kind not in "ui"
            or instances.min() < 0
            or classes.min() < 0
            or instances.max() > np.iinfo(np.int32).max
            or classes.max() > 6
            or np.any(classes[instances == 0] != 0)
        ):
            raise ValueError(
                "Expected nonnegative instance IDs and nucleus classes 0 through 6."
            )

        # Crop before creating flow targets so they describe the cropped objects.
        top, left = (size // 2 - 128 for size in instances.shape)
        crop = np.s_[top : top + 256, left : left + 256]
        samples.append(
            (
                offset,
                split,
                patient,
                {key: value[crop] for key, value in arrays.items()},
            )
        )

    if patients["train"] & patients["validation"]:
        raise ValueError("Training and validation contain the same patient.")
    destination.mkdir(parents=True)
    for split in patients:
        (destination / split / "images").mkdir(parents=True)
        (destination / split / "labels").mkdir()
    records = []
    for offset, split, patient, arrays in samples:
        name = f"sample_{offset:04d}"
        Image.fromarray(arrays["image"]).save(
            destination / split / "images" / f"{name}.png"
        )
        savemat(
            destination / split / "labels" / f"{name}.mat",
            {"inst": arrays["inst"], "type": arrays["type"]},
        )
        records.append({"row": offset, "split": split, "patient": patient})
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "png_sha256": PNG_SHA256,
                "samples": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare_panoptils(parser.parse_args().destination)
    print("Prepared 3 training regions and 1 validation region (256 × 256).")
