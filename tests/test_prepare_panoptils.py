"""Preparation keeps patients independent and image/instance/type patches aligned."""

import hashlib
import json
from io import BytesIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from scipy.io import loadmat

from tools import prepare_panoptils as preparation


def _png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    instances = np.zeros((512, 512), dtype=np.uint16)
    instances[20:90, 30:100] = 11
    instances[300:390, 320:400] = 22
    classes = np.where(instances == 11, 3, np.where(instances == 22, 5, 0)).astype(
        np.uint8
    )
    rows_grid, cols_grid = np.mgrid[:512, :512]
    image = np.stack((classes * 40, rows_grid % 256, cols_grid % 256), axis=-1).astype(
        np.uint8
    )
    arrays = {"image": image, "inst": instances, "type": classes}
    rows = [
        dict(
            slide_name=f"TCGA-AA-{i:04d}-DX1",
            sem=b"unused, not PNG",
            **{key: _png(array) for key, array in arrays.items()},
        )
        for i in range(5)
    ]
    rows.append({**rows[0], "slide_name": "TCGA-AA-0000-DX2"})
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    monkeypatch.setattr(
        preparation, "PARQUET_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return source, rows, arrays


def test_prepares_all_aligned_patches_after_patient_split(tmp_path, archive):
    source, rows, arrays = archive
    destination = tmp_path / "prepared"
    preparation.prepare_panoptils(source, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    patients = {
        split: {row["patient"] for row in manifest["samples"] if row["split"] == split}
        for split in ("train", "validation")
    }
    assert patients["train"].isdisjoint(patients["validation"])
    assert len(patients["train"]) == 4 and len(patients["validation"]) == 1
    assert sum(split["patches"] for split in manifest["splits"].values()) == 24
    for record in manifest["samples"]:
        split = record["split"]
        for y in (0, 256):
            for x in (0, 256):
                name = f"{record['file']}_y{y:04d}_x{x:04d}"
                image = np.asarray(
                    Image.open(destination / split / "images" / f"{name}.png")
                )
                masks = loadmat(destination / split / "labels" / f"{name}.mat")
                area = np.s_[y : y + 256, x : x + 256]
                np.testing.assert_array_equal(image, arrays["image"][area])
                np.testing.assert_array_equal(masks["inst"], arrays["inst"][area])
                np.testing.assert_array_equal(masks["type"], arrays["type"][area])
                assert "sem" not in masks
    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        preparation.prepare_panoptils(source, destination)
    assert (destination / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("defect", ["digest", "class", "shape", "patients"])
def test_rejects_invalid_inputs_without_publishing_partial_data(
    tmp_path, archive, monkeypatch, defect
):
    source, rows, arrays = archive
    if defect == "digest":
        source.write_bytes(source.read_bytes() + b"changed")
    else:
        if defect == "class":
            invalid = arrays["type"].copy()
            invalid[arrays["inst"] > 0] = 7
            rows[-1]["type"] = _png(invalid)
        elif defect == "shape":
            rows[-1]["inst"] = _png(arrays["inst"][:-1])
        else:
            for row in rows:
                row["slide_name"] = "TCGA-AA-0000-DX1"
        pq.write_table(pa.Table.from_pylist(rows), source)
        monkeypatch.setattr(
            preparation,
            "PARQUET_SHA256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError):
        preparation.prepare_panoptils(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".panoptils-*"))
