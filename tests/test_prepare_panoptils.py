"""Execute preparation notebook cells against small, offline parquet fixtures."""

import hashlib
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from scipy.io import loadmat


def _png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def notebook_cells():
    path = Path(__file__).resolve().parents[1] / "docs/tutorials/prepare-data.ipynb"
    notebook = json.loads(path.read_text())
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture
def archive(tmp_path):
    instances = np.zeros((512, 512), dtype=np.uint16)
    instances[220:300, 220:300] = 11
    instances[300:390, 320:400] = 22
    instances[20:30, 30:40] = 33
    classes = np.where(instances == 22, 5, np.where(instances > 0, 3, 0)).astype(
        np.uint8
    )
    classes[220, 220:300] = 0
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
    return source, rows, arrays


@pytest.fixture
def preparation(tmp_path, archive, notebook_cells, monkeypatch):
    source, _, _ = archive
    namespace = {}
    exec(notebook_cells[0], namespace)
    namespace["data_root"] = tmp_path / "prepared"
    namespace["archive_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()

    def download(**kwargs):
        assert kwargs == {
            "repo_id": "histolytics-hub/panoptils_refined",
            "repo_type": "dataset",
            "filename": "panoptils_refined.parquet",
            "revision": "d40293bc1ac0d0da8f271959fdb990aeea2d884f",
        }
        return str(source)

    namespace["hf_hub_download"] = download
    monkeypatch.setattr(plt, "show", lambda: None)
    yield namespace
    plt.close("all")


def _prepare(namespace, cells):
    for source in cells[1:-2]:
        exec(source, namespace)


def test_notebook_prepares_aligned_patches_and_counts_original_nuclei(
    archive, preparation, notebook_cells
):
    _, _, arrays = archive
    _prepare(preparation, notebook_cells)
    destination = preparation["data_root"]
    manifest = json.loads((destination / "manifest.json").read_text())
    patients = {
        split: {row["patient"] for row in manifest["samples"] if row["split"] == split}
        for split in ("train", "validation")
    }
    assert patients["train"].isdisjoint(patients["validation"])
    assert len(patients["train"]) == 4 and len(patients["validation"]) == 1
    assert sum(split["patches"] for split in manifest["splits"].values()) == 24
    for count in manifest["splits"].values():
        assert count["class_instances"] == [
            0,
            0,
            0,
            2 * count["regions"],
            0,
            count["regions"],
            0,
        ]
    for record in manifest["samples"]:
        for y in (0, 256):
            for x in (0, 256):
                name = f"{record['file']}_y{y:04d}_x{x:04d}"
                folder = destination / record["split"]
                with Image.open(folder / "images" / f"{name}.png") as image:
                    image = np.array(image)
                masks = loadmat(folder / "labels" / f"{name}.mat")
                area = np.s_[y : y + 256, x : x + 256]
                np.testing.assert_array_equal(image, arrays["image"][area])
                np.testing.assert_array_equal(masks["inst"], arrays["inst"][area])
                np.testing.assert_array_equal(masks["type"], arrays["type"][area])
                assert "sem" not in masks
        if record["split"] == "validation":
            masks = loadmat(
                destination / "validation/regions/labels" / f"{record['file']}.mat"
            )
            np.testing.assert_array_equal(masks["inst"], arrays["inst"])
            np.testing.assert_array_equal(masks["type"], arrays["type"])

    for source in notebook_cells[-2:]:
        exec(source, preparation)
    table = preparation["instance_counts"]
    assert table.loc["inflammatory"].sum() == 12
    assert table.loc["other"].sum() == 6
    assert table.to_numpy().sum() == 18
    assert "background" not in table.index
    assert len(preparation["fig"].axes) == 3

    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _prepare(preparation, notebook_cells)
    assert (destination / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    "defect", ["digest", "class", "shape", "patients", "background", "rgb"]
)
def test_rejects_invalid_inputs_without_partial_dataset(
    tmp_path, archive, preparation, notebook_cells, defect
):
    source, rows, arrays = archive
    if defect == "digest":
        source.write_bytes(source.read_bytes() + b"changed")
    else:
        if defect in ("class", "background"):
            invalid = arrays["type"].copy()
            if defect == "class":
                invalid[arrays["inst"] > 0] = 7
            else:
                invalid[0, 0] = 3
            rows[-1]["type"] = _png(invalid)
        elif defect == "shape":
            rows[-1]["inst"] = _png(arrays["inst"][:-1])
        elif defect == "rgb":
            rows[-1]["image"] = _png(arrays["image"][..., 0])
        else:
            for row in rows:
                row["slide_name"] = "TCGA-AA-0000-DX1"
        pq.write_table(pa.Table.from_pylist(rows), source)
        preparation["archive_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        _prepare(preparation, notebook_cells)
    assert not preparation["data_root"].exists()
    assert not list(tmp_path.glob(".panoptils-*"))


@pytest.mark.parametrize(
    "instances,classes,expected",
    [
        (
            [[70001, 70001, 70001], [22, 22, 22]],
            [[0, 3, 3], [0, 0, 0]],
            [1, 0, 0, 1, 0, 0, 0],
        ),
        ([[11, 11, 11]], [[3, 5, 5]], [0, 0, 0, 0, 0, 1, 0]),
        ([[0, 0]], [[0, 0]], [0, 0, 0, 0, 0, 0, 0]),
    ],
)
def test_counts_each_nucleus_once_by_nonbackground_majority(
    preparation, notebook_cells, instances, classes, expected
):
    for source in notebook_cells[1:-2]:
        if source.startswith("def count_nuclei"):
            exec(source, preparation)
    actual = preparation["count_nuclei"](np.array(instances), np.array(classes))
    np.testing.assert_array_equal(actual, expected)
