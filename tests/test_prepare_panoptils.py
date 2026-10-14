"""Offline checks for the tutorial dataset preparation boundary."""

import base64
import hashlib
from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from scipy.io import loadmat

from tools import prepare_panoptils as preparation


@pytest.fixture
def dataset_rows(monkeypatch):
    instances = np.zeros((260, 260), dtype=np.uint16)
    instances[20:80, 30:90] = 11
    instances[150:200, 160:210] = 22
    classes = np.zeros_like(instances, dtype=np.uint8)
    classes[instances == 11] = 3
    classes[instances == 22] = 5
    image = np.zeros((260, 260, 3), dtype=np.uint8)
    image[..., 0] = classes * 40
    arrays = {"image": image, "inst": instances, "type": classes}
    encoded, hashes = {}, {}
    for key, value in arrays.items():
        buffer = BytesIO()
        Image.fromarray(value).save(buffer, format="PNG")
        encoded[key] = base64.b64encode(buffer.getvalue()).decode()
        hashes[key] = hashlib.sha256(buffer.getvalue()).hexdigest()
    patients = {
        0: "TCGA-AA-0001",
        1: "TCGA-AA-0001",
        24: "TCGA-AA-0002",
        25: "TCGA-AA-0003",
    }
    rows = {
        offset: {
            **encoded,
            "slide_name": f"{patients[offset]}-DX1",
            "sem": "not a PNG: this must never be decoded",
        }
        for offset in preparation.SELECTION
    }
    monkeypatch.setattr(
        preparation, "PNG_SHA256", {offset: dict(hashes) for offset in rows}
    )
    responses = []
    for row in rows.values():
        response = Mock()
        response.json.return_value = {"rows": [{"row": row}]}
        responses.append(response)
    getter = Mock(side_effect=responses)
    monkeypatch.setattr(preparation.requests, "get", getter)
    return rows, arrays, getter


def test_prepares_aligned_nuclei_only_data(tmp_path, dataset_rows):
    _, arrays, getter = dataset_rows
    destination = tmp_path / "panoptils"
    preparation.prepare_panoptils(destination)
    assert getter.call_count == 4
    assert len(list((destination / "train/images").glob("*.png"))) == 3
    assert len(list((destination / "validation/images").glob("*.png"))) == 1
    image = np.asarray(Image.open(destination / "train/images/sample_0000.png"))
    labels = loadmat(destination / "train/labels/sample_0000.mat")
    assert {key for key in labels if not key.startswith("__")} == {"inst", "type"}
    np.testing.assert_array_equal(image, arrays["image"][2:258, 2:258])
    for key in ("inst", "type"):
        np.testing.assert_array_equal(labels[key], arrays[key][2:258, 2:258])
    assert set(np.unique(labels["inst"])) == {0, 11, 22}
    assert set(np.unique(labels["type"])) == {0, 3, 5}
    with pytest.raises(FileExistsError):
        preparation.prepare_panoptils(destination)
    assert getter.call_count == 4


@pytest.mark.parametrize(
    "invalid", ["changed_bytes", "patient_overlap", "class_range", "misaligned"]
)
def test_rejects_invalid_data_before_writing(tmp_path, dataset_rows, invalid):
    rows, _, _ = dataset_rows
    if invalid == "changed_bytes":
        rows[0]["inst"] = rows[0]["image"]
    elif invalid == "patient_overlap":
        rows[25]["slide_name"] = rows[0]["slide_name"]
    else:
        array = np.full((260, 260), 7, dtype=np.uint8)
        if invalid == "misaligned":
            array = array[:259]
        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        rows[0]["type"] = base64.b64encode(buffer.getvalue()).decode()
        preparation.PNG_SHA256[0]["type"] = hashlib.sha256(
            buffer.getvalue()
        ).hexdigest()
    destination = tmp_path / "panoptils"
    with pytest.raises(ValueError):
        preparation.prepare_panoptils(destination)
    assert not destination.exists()
