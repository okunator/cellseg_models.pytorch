"""Execute API reference examples with local files and no downloaded weights."""

import ast
import doctest
import inspect
import json
import pydoc
import random
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import loadmat, savemat

API = Path(__file__).resolve().parents[1] / "docs" / "api"


def _python_blocks(page: Path) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", page.read_text(), re.M | re.S)


def _run_page(name: str) -> dict:
    namespace = {}
    blocks = _python_blocks(API / f"{name}.md")
    assert blocks, f"No Python examples found in {name}"
    for index, source in enumerate(blocks):
        exec(compile(source, f"{name}.md:block-{index + 1}", "exec"), namespace)
    return namespace


def test_api_example_coverage():
    pages = {page.stem for page in API.glob("*.md") if _python_blocks(page)}
    assert pages == {
        "models",
        "model-networks",
        "metrics",
        "losses",
        "transforms",
        "inference",
        "datasets",
        "file-io",
        "export",
        "slide-reading",
        "wsi-inference",
    }


def test_guide_imports_have_api_entries():
    # shortcut: checks explicit from-imports, extend for module-alias guide examples.
    pytest.importorskip("albumentations")
    objects = [
        pydoc.locate(target)
        for page in API.glob("*.md")
        for target in re.findall(
            r"^\.\. auto(?:class|function|method):: ([\w.]+)", page.read_text(), re.M
        )
    ]
    assert objects and all(obj is not None for obj in objects)
    documented = {id(obj) for obj in objects}
    root = API.parents[1]
    sources = []
    for path in (
        root / "README.md",
        root / "docs/quickstart.md",
        root / "docs/user-guide.md",
    ):
        sources.extend(_python_blocks(path))
    for path in (root / "docs/tutorials").glob("*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            source = "".join(cell["source"])
            sources.extend(
                [source]
                if cell["cell_type"] == "code"
                else re.findall(r"^```python\n(.*?)^```", source, re.M | re.S)
            )
    imports = 0
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "cellseg_models_pytorch"
            ):
                for alias in node.names:
                    target = f"{node.module}.{alias.name}"
                    assert id(pydoc.locate(target)) in documented, target
                    imports += 1
    assert imports > 0
    print(f"{imports} guide import occurrences resolve to documented objects")


@pytest.fixture(autouse=True)
def example_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    image = np.arange(64 * 64 * 3).reshape(64, 64, 3).astype(np.uint8)
    instances = np.zeros((64, 64), dtype=np.int32)
    instances[16:48, 16:48] = 1
    labels = {"inst": instances, "type": instances * 2}
    for directory in ("data/train/images", "data/train/labels"):
        Path(directory).mkdir(parents=True)
    Image.fromarray(image).save("data/image.png")
    Image.fromarray(image).save("data/train/images/patch.png")
    savemat("data/labels.mat", labels)
    savemat("data/train/labels/patch.mat", labels)
    try:
        yield image, labels
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)


@pytest.fixture
def local_checkpoint():
    from safetensors.torch import save_model

    from cellseg_models_pytorch.models.cellpose import CellPose

    model = CellPose(
        3, enc_name="resnet18", enc_pretrain=False, device=torch.device("cpu")
    )
    path = Path("checkpoints/cellpose.safetensors")
    path.parent.mkdir()
    save_model(model.model, path)
    return model


@pytest.mark.parametrize(
    "name", ["models", "model-networks", "metrics", "losses", "transforms"]
)
def test_api_page_examples(name):
    if name == "transforms":
        pytest.importorskip("albumentations")
    ns = _run_page(name)
    if name == "model-networks":
        output = ns["output"]
        assert output.type_map.shape == (1, 3, 64, 64)
        assert output.aux_map.shape == (1, 2, 64, 64)
        assert torch.isfinite(output.type_map).all()
    elif name == "metrics":
        assert ns["score"]["pq"] == pytest.approx(1)
        assert ns["foreground_iou"] == pytest.approx(1)
    elif name == "losses":
        assert ns["loss"].ndim == 0 and torch.isfinite(ns["loss"])
        for key in ("flow_prediction", "class_logits"):
            gradient = ns[key].grad
            assert gradient is not None and torch.isfinite(gradient).all()
    elif name == "transforms":
        np.testing.assert_array_equal(ns["result"]["masks"][0], ns["instances"])
        assert ns["result"]["image"].dtype == np.float32
        assert ns["normalized_image"].min() == pytest.approx(0, abs=1e-7)
        assert ns["normalized_image"].max() == pytest.approx(1)
        expected = np.zeros((8, 8, 1), dtype=np.uint8)
        expected[2:6, 2:6] = 1
        np.testing.assert_array_equal(ns["targets"][0], expected)
        assert ns["position_target"].shape == (2, 16, 16)
        assert ns["ray_target"].shape == (8, 16, 16)


def test_api_inference_example(local_checkpoint):
    ns = _run_page("inference")
    for key in ("instances", "classes"):
        mask = ns[key]
        assert mask.shape == (64, 64) and mask.dtype == np.int32
    assert ns["prediction"]["nuc"].type_map.shape == (1, 64, 64)
    assert torch.isfinite(ns["prediction"]["nuc"].aux_map).all()
    assert ns["model"].inference_mode
    assert not ns["model"].model.training


@pytest.fixture
def slide_backend(monkeypatch):
    """Use the real OpenSlide adapter with an in-memory native-reader substitute."""
    from types import SimpleNamespace

    from cellseg_models_pytorch.wsi import openslide_reader

    canvas = np.full((256, 256, 3), 254, dtype=np.uint8)
    for y in (32, 160):
        for x in (32, 160):
            canvas[y : y + 64, x : x + 64] = (150, 70, 110)
    reads = []

    def read_region(location, level, size):
        reads.append((location, level, size))
        x, y = location
        width, height = size
        downsample = 2**level
        return (
            Image.fromarray(canvas)
            .convert("RGBA")
            .crop((x, y, x + width * downsample, y + height * downsample))
            .resize(size)
        )

    native = SimpleNamespace(
        level_dimensions=((256, 256), (128, 128)),
        properties={"openslide.mpp-y": "0.25", "openslide.mpp-x": "0.25"},
        get_thumbnail=lambda size: Image.fromarray(canvas).resize(size),
        read_region=read_region,
    )
    monkeypatch.setattr(openslide_reader, "HAS_OPENSLIDE", True)
    monkeypatch.setattr(
        openslide_reader,
        "openslide",
        SimpleNamespace(OpenSlide=lambda path: native),
        raising=False,
    )
    Path("data/slide.svs").touch()
    return reads


def test_api_slide_reading_example(slide_backend):
    pytest.importorskip("matplotlib")
    ns = _run_page("slide-reading")
    assert ns["reader"].dimensions == (256, 256)
    assert ns["reader"].pixel_size == (0.25, 0.25)
    assert ns["tile"].shape == (256, 256, 3)
    assert ns["tile"].dtype == np.uint8
    assert ns["preview"].size == (256, 256)
    assert ns["properties"] == {
        "num_tiles": 1,
        "level": 0,
        "level_downsample": (1.0, 1.0),
        "width": 256,
        "height": 256,
        "overlap": 0.25,
        "max_background": 0.95,
    }
    assert len(ns["named_spots"]) == 4
    assert {bounds for _, bounds in ns["named_spots"]} == {
        (x, y, 64, 64) for x in (32, 160) for y in (32, 160)
    }
    assert slide_backend == [((0, 0), 0, (256, 256))]


def test_api_wsi_inference_example(slide_backend, local_checkpoint):
    pytest.importorskip("albumentations")
    import geopandas as gpd

    ns = _run_page("wsi-inference")
    nuclei = ns["nuclei"]
    assert {"geometry", "class_name", "uid"} <= set(nuclei.columns)
    assert nuclei.uid.is_unique and nuclei.geometry.is_valid.all()
    assert slide_backend == [((0, 0), 0, (256, 256))]
    merged = ns["merged"]
    assert len(merged) == 2
    assert merged.class_name.tolist() == ["tumor", "immune"]
    assert merged.geometry.equals(
        gpd.GeoSeries(ns["predictions"].geometry.iloc[[0, 2]]).reset_index(drop=True)
    )
    assert merged.uid.tolist() == [0, 1]
    assert ns["filtered"].equals(merged)


@pytest.mark.optional
@pytest.mark.parametrize("name", ["datasets", "file-io"])
def test_api_hdf5_examples(name, example_files):
    tables = pytest.importorskip("tables")
    pytest.importorskip("albumentations")
    image, labels = example_files
    with tables.open_file("data/train.h5", "w") as handle:
        handle.create_array("/", "image", image[None])
        for key, mask in labels.items():
            handle.create_array("/", key, mask[None])
    ns = _run_page(name)
    if name == "datasets":
        assert ns["sample"]["image"].shape == (3, 64, 64)
        assert ns["sample"]["cellpose_inst"].shape == (2, 64, 64)
        assert ns["h5_sample"]["inst"].dtype == torch.int64
        torch.testing.assert_close(
            ns["h5_sample"]["inst_cellpose"], ns["sample"]["cellpose_inst"].float()
        )
    else:
        saved = loadmat("outputs/prediction.mat")
        for key, expected in labels.items():
            np.testing.assert_array_equal(saved[key], expected)
            np.testing.assert_array_equal(ns["sample"][key], expected)
        np.testing.assert_array_equal(ns["sample"]["image"], image)


@pytest.mark.optional
def test_api_export_example(local_checkpoint):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    runtime = pytest.importorskip("onnxruntime")
    from cellseg_models_pytorch.models.cellpose.cellpose_unet import omnipose_nuclei

    network = omnipose_nuclei(3, enc_name="resnet18", enc_pretrain=False)
    torch.save(network.state_dict(), "checkpoints/omnipose.pt")
    ns = _run_page("export")
    for path, model in (
        (ns["output_path"], local_checkpoint.model),
        (ns["omnipose_path"], ns["network"]),
    ):
        onnx.checker.check_model(onnx.load(path))
        session = runtime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        assert [output.name for output in session.get_outputs()] == [
            "flow_map",
            "type_map",
        ]
        model.eval()
        for batch in (1, 2):
            image = torch.rand(batch, 3, 256, 256)
            with torch.inference_mode():
                expected = model(image)["nuc"]
            actual = session.run(None, {"image": image.numpy()})
            for result, reference in zip(actual, (expected.aux_map, expected.type_map)):
                assert np.isfinite(result).all()
                np.testing.assert_allclose(
                    result, reference.numpy(), rtol=2e-4, atol=5e-5
                )


def test_api_docstring_examples(local_checkpoint):
    """Include class constructors and explicitly listed methods from every page."""
    pytest.importorskip("albumentations")
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner()
    seen = set()
    rst_count = 0
    for page in API.glob("*.md"):
        for kind, target, members in re.findall(
            r"^\.\. auto(class|function|method):: ([\w.]+)[^\n]*\n(?:   :members: ([^\n]+)\n)?",
            page.read_text(),
            re.M,
        ):
            obj = pydoc.locate(target)
            assert obj is not None, target
            objects = [obj]
            if kind == "class":
                objects += [obj.__init__]
                objects += [
                    getattr(obj, member.strip())
                    for member in members.split(",")
                    if member
                ]
            for item in objects:
                doc = inspect.getdoc(item) or ""
                if doc in seen:
                    continue
                seen.add(doc)
                if ">>>" not in doc and ".. code-block:: python" not in doc:
                    continue
                globs = vars(inspect.getmodule(item)).copy()
                test = parser.get_doctest(doc, globs, target, str(page), 0)
                runner.run(test)
                for source in re.findall(
                    r"^([ ]*)\.\. code-block:: python\n\n((?:(?:\1    .*)?\n)+)",
                    doc,
                    re.M,
                ):
                    namespace = {}
                    exec(textwrap.dedent(source[1]), namespace)
                    restored = namespace["model"]
                    assert restored.device == torch.device("cpu")
                    for key, value in local_checkpoint.model.state_dict().items():
                        torch.testing.assert_close(
                            restored.model.state_dict()[key], value
                        )
                    rst_count += 1
    result = runner.summarize()
    assert result.failed == 0
    assert result.attempted > 0 and rst_count == 1
    print(
        f"{result.attempted} doctest statements and {rst_count} docstring code block passed"
    )
