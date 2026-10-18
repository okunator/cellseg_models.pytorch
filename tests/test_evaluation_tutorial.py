"""Check the evaluation notebook against independently specified mask scores."""

import ast
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.io import savemat
from skimage.color import label2rgb

NOTEBOOK = Path(__file__).resolve().parents[1] / "docs/tutorials/evaluation.ipynb"


def _cells() -> dict[str, str]:
    notebook = json.loads(NOTEBOOK.read_text())
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


@pytest.fixture
def dataframe_display():
    pl = pytest.importorskip("polars")
    with pl.Config():
        yield


@pytest.fixture
def scoring(dataframe_display):
    cells = _cells()
    namespace = {}
    exec(cells["metrics-setup"], namespace)
    for name in ("metrics-compute", "metrics-overlays"):
        tree = ast.parse(cells[name])
        definitions = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
            type_ignores=[],
        )
        exec(compile(definitions, str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("reference", "prediction", "pq", "dq", "sq", "dice", "iou", "counts"),
    [
        ([[11, 11]], [[103, 103]], 1, 1, 1, 1, 1, (1, 1)),
        (
            [[11, 11, 0, 29, 29]],
            [[103, 103, 0, 0, 0]],
            2 / 3,
            2 / 3,
            1,
            2 / 3,
            0.5,
            (2, 1),
        ),
        (
            [[11, 11, 0, 0, 0]],
            [[103, 103, 0, 907, 907]],
            2 / 3,
            2 / 3,
            1,
            2 / 3,
            0.5,
            (1, 2),
        ),
        ([[11, 11, 11, 11]], [[103, 103, 907, 907]], 0, 0, 0, 1, 1, (1, 2)),
        ([[11, 11, 29, 29]], [[103, 103, 103, 103]], 0, 0, 0, 1, 1, (2, 1)),
        ([[11, 11, 0, 0]], [[0, 0, 103, 103]], 0, 0, 0, 0, 0, (1, 1)),
        ([[11, 11, 11, 11]], [[103, 103, 103, 0]], 0.75, 1, 0.75, 6 / 7, 0.75, (1, 1)),
        ([[0, 0]], [[103, 103]], 0, 0, 0, 0, 0, (0, 1)),
        ([[11, 11]], [[0, 0]], 0, 0, 0, 0, 0, (1, 0)),
    ],
    ids=[
        "different-ids-no-background",
        "missed",
        "extra",
        "split-at-half-iou",
        "merged-at-half-iou",
        "disjoint",
        "partial-match",
        "only-predictions",
        "only-reference",
    ],
)
def test_notebook_scores_known_mask_pairs(
    scoring, reference, prediction, pq, dq, sq, dice, iou, counts
):
    reference = np.array(reference, dtype=np.int32)
    prediction = np.array(prediction, dtype=np.int32)
    before = (reference.copy(), prediction.copy())

    actual = scoring["score_masks"](reference, prediction).row(0, named=True)

    assert actual == pytest.approx(
        {
            "Reference nuclei": counts[0],
            "Predicted nuclei": counts[1],
            "PQ": pq,
            "DQ": dq,
            "SQ": sq,
            "Dice": dice,
            "IoU": iou,
        }
    )
    np.testing.assert_array_equal(reference, before[0])
    np.testing.assert_array_equal(prediction, before[1])


def test_empty_pair_is_undefined_instead_of_rewarded_or_averaged_as_zero(scoring):
    empty = np.zeros((2, 3), dtype=np.int32)
    assert scoring["score_masks"](empty, empty).row(0, named=True) == {
        "Reference nuclei": 0,
        "Predicted nuclei": 0,
        "PQ": None,
        "DQ": None,
        "SQ": None,
        "Dice": None,
        "IoU": None,
    }


def test_inference_save_example_pairs_masks_by_image_name_without_overwriting(
    scoring, tmp_path, monkeypatch
):
    notebook = json.loads(NOTEBOOK.with_name("inference.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("id") == "2fe2c5f7"
    )
    instances = np.array([[0, 11, 11]], dtype=np.int32)
    classes = np.array([[0, 3, 3]], dtype=np.int32)
    namespace = {
        "Path": Path,
        "np": np,
        "image_path": Path("heldout.png"),
        "instances": instances,
        "classes": classes,
    }
    monkeypatch.chdir(tmp_path)
    exec(source, namespace)
    start = source.index("# output_path.parent.mkdir")
    save = "\n".join(line.removeprefix("# ") for line in source[start:].splitlines())
    exec(save, namespace)
    path = tmp_path / "predictions/heldout.npz"
    saved = path.read_bytes()
    actual_instances, actual_classes = scoring["load_prediction"](path)
    np.testing.assert_array_equal(actual_instances, instances)
    np.testing.assert_array_equal(actual_classes, classes)
    with pytest.raises(FileExistsError):
        exec(save, namespace)
    assert path.read_bytes() == saved


def test_wrong_type_is_a_miss_and_an_extra_while_untyped_nuclei_remain_foreground(
    scoring, tmp_path
):
    reference = np.array([[11, 11, 0, 29, 29]], dtype=np.int32)
    prediction = np.array([[103, 103, 0, 907, 907]], dtype=np.int32)
    reference_types = np.array([[1, 1, 0, 0, 0]], dtype=np.int32)
    predicted_types = np.array([[2, 2, 0, 0, 0]], dtype=np.int32)
    labels = tmp_path / "validation/regions/labels"
    labels.mkdir(parents=True)
    savemat(labels / "example.mat", {"inst": reference, "type": reference_types})
    prediction_path = tmp_path / "example.npz"
    np.savez_compressed(prediction_path, instances=prediction, classes=predicted_types)
    scoring.update(data_root=tmp_path, prediction_root=tmp_path)

    exec(_cells()["metrics-images"], scoring)

    table = scoring["scores"].filter(scoring["pl"].col("Cell type") != "all nuclei")
    by_type = {row["Cell type"]: row for row in table.to_dicts()}
    overall = scoring["image_scores"].row(0, named=True)
    assert overall["PQ"] == pytest.approx(1)
    assert overall["Dice"] == pytest.approx(1)
    assert overall["Reference nuclei"] == overall["Predicted nuclei"] == 2
    assert by_type["neoplastic"]["PQ"] == by_type["stromal"]["PQ"] == 0
    assert by_type["neoplastic"]["Reference nuclei"] == 1
    assert by_type["neoplastic"]["Predicted nuclei"] == 0
    assert by_type["stromal"]["Reference nuclei"] == 0
    assert by_type["stromal"]["Predicted nuclei"] == 1
    assert by_type["inflammatory"]["PQ"] is None
    assert len(by_type) == 6 and "untyped" not in by_type


def test_overlays_preserve_tissue_rgb_and_untyped_nuclei_and_touching_boundaries(
    scoring,
):
    image = np.full((8, 18, 3), [80, 120, 160], dtype=np.uint8)
    instances = np.zeros((8, 18), dtype=np.int32)
    instances[1:7, 1:6] = 11
    instances[1:7, 6:11] = 29
    instances[2:6, 12:16] = 103
    classes = np.where((instances == 11) | (instances == 29), 2, 0)
    original = (image.copy(), instances.copy(), classes.copy())

    actual = scoring["cell_type_overlay"](image, instances, classes)

    assert actual.shape == image.shape and actual.dtype == np.float32
    base = image.astype(np.float32) / 255
    palette = scoring["class_palette"]
    np.testing.assert_allclose(actual[0, 0], base[0, 0])
    np.testing.assert_allclose(actual[3, 3], 0.75 * base[3, 3] + 0.25 * palette[2])
    # Both sides of a shared boundary stay visible even with the same cell type.
    np.testing.assert_array_equal(actual[3, 5], palette[2])
    np.testing.assert_array_equal(actual[3, 6], palette[2])
    np.testing.assert_array_equal(actual[2, 12], palette[0])
    np.testing.assert_allclose(actual[3, 13], 0.75 * base[3, 13] + 0.25 * palette[0])
    for value, before in zip((image, instances, classes), original):
        np.testing.assert_array_equal(value, before)


def test_notebook_loads_saved_predictions_and_averages_defined_image_scores(
    scoring, tmp_path, monkeypatch
):
    directory = tmp_path / "docs/tutorials"
    directory.mkdir(parents=True)
    predictions = directory / "predictions"
    predictions.mkdir()
    data = tmp_path / "data/panoptils-nuclei/validation/regions"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    first = np.zeros((6, 12), dtype=np.int32)
    first[1:3, 1:3] = 11
    first[1:3, 7:9] = 29
    missed = np.where(first == 11, 103, 0).astype(np.int32)
    second = np.zeros_like(first)
    second[2:4, 4:6] = 51
    perfect = np.where(second > 0, 907, 0).astype(np.int32)
    blank = np.zeros_like(first)
    for name, reference, prediction, cell_type in (
        ("a", first, missed, 1),
        ("b", second, perfect, 3),
        ("c", blank, blank, 1),
    ):
        savemat(
            data / "labels" / f"{name}.mat",
            {
                "inst": reference,
                "type": (reference > 0).astype(np.int32) * cell_type,
            },
        )
        np.savez_compressed(
            predictions / f"{name}.npz",
            instances=prediction,
            classes=(prediction > 0).astype(np.int32) * cell_type,
        )
        assert cv2.imwrite(
            str(data / "images" / f"{name}.png"),
            np.full((6, 12, 3), 128, dtype=np.uint8),
        )
    monkeypatch.chdir(directory)
    monkeypatch.setattr(plt, "show", Mock())
    namespace = {}
    try:
        for identifier, source in _cells().items():
            exec(source, namespace)
            if identifier == "metrics-setup":
                namespace["example_images"] = ["a", "b"]
        summary = namespace["summary"].row(0, named=True)
        assert summary == pytest.approx(
            {
                "Images": 3,
                "Scored images": 2,
                "Reference nuclei": 3,
                "Predicted nuclei": 2,
                "PQ": 5 / 6,
                "DQ": 5 / 6,
                "SQ": 1,
                "Dice": 5 / 6,
                "IoU": 0.75,
            }
        )
        by_type = {
            row["Cell type"]: row for row in namespace["class_summary"].to_dicts()
        }
        assert by_type["neoplastic"]["Scored images"] == 1
        assert by_type["neoplastic"]["PQ"] == pytest.approx(2 / 3)
        assert by_type["inflammatory"]["Scored images"] == 1
        assert by_type["inflammatory"]["PQ"] == pytest.approx(1)
        assert by_type["stromal"]["Scored images"] == 0
        assert by_type["stromal"]["PQ"] is None
        assert namespace["image_scores"].height == 3
        figure = namespace["fig"]
        assert [axis.get_title() for axis in figure.axes] == [
            "a: image",
            "Reference nuclei",
            "Predicted nuclei",
            "b: image",
            "Reference nuclei",
            "Predicted nuclei",
        ]
        np.testing.assert_array_equal(
            figure.axes[1].images[0].get_array(), label2rgb(first, bg_label=0)
        )
        np.testing.assert_array_equal(
            figure.axes[2].images[0].get_array(), label2rgb(missed, bg_label=0)
        )
        np.testing.assert_array_equal(
            figure.axes[4].images[0].get_array(), label2rgb(second, bg_label=0)
        )
        np.testing.assert_array_equal(
            figure.axes[5].images[0].get_array(), label2rgb(perfect, bg_label=0)
        )
        assert namespace["class_scores"]["Image"].unique(
            maintain_order=True
        ).to_list() == ["a", "b"]
        assert namespace["class_scores"].height == 12
        overlays = namespace["overlay_fig"]
        assert [axis.get_title() for axis in overlays.axes] == [
            "a: image",
            "Reference cell types",
            "Predicted cell types",
            "b: image",
            "Reference cell types",
            "Predicted cell types",
        ]
        palette = namespace["class_palette"]
        np.testing.assert_array_equal(
            overlays.axes[1].images[0].get_array()[1, 1], palette[1]
        )
        np.testing.assert_array_equal(
            overlays.axes[2].images[0].get_array()[1, 1], palette[1]
        )
        np.testing.assert_allclose(
            overlays.axes[2].images[0].get_array()[1, 7], [128 / 255] * 3
        )
        np.testing.assert_array_equal(
            overlays.axes[5].images[0].get_array()[2, 4], palette[3]
        )
        assert [
            text.get_text() for text in overlays.legends[0].get_texts()
        ] == namespace["class_names"]
        assert plt.show.call_count == 2
        with np.load(predictions / "a.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["instances"], missed)
    finally:
        plt.close("all")
