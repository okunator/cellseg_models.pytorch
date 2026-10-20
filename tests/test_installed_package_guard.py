"""Check installed-package OpenCV providers and bundled license material."""

import sys
from importlib.metadata import Distribution
from pathlib import Path

import pytest

from tools import check_installed_package as guard


@pytest.mark.parametrize(
    ("extra", "installed", "error"),
    [
        (None, [], None),
        ("training", ["albumentations", "tables"], None),
        ("onnx", ["onnx", "onnxscript"], None),
        ("openslide", ["openslide", "openslide_bin"], None),
        ("training", ["albumentations"], "Missing optional package tables"),
        (
            "onnx",
            ["onnx", "onnxscript", "tables"],
            "Unexpected optional package tables",
        ),
    ],
)
def test_optional_install_boundaries(monkeypatch, extra, installed, error):
    monkeypatch.setattr(
        guard, "find_spec", lambda name: object() if name in installed else None
    )
    if error is None:
        guard.check_optional_dependencies(extra)
    else:
        with pytest.raises(RuntimeError, match=error):
            guard.check_optional_dependencies(extra)


@pytest.mark.parametrize("providers", [[], ["opencv-python", "opencv-python-headless"]])
def test_installed_gate_requires_one_opencv_provider(monkeypatch, providers):
    monkeypatch.setattr(
        guard.cellseg_models_pytorch,
        "__file__",
        str(Path(sys.prefix) / "lib/cellseg_models_pytorch/__init__.py"),
    )
    monkeypatch.setattr(
        guard, "version", lambda name: guard.cellseg_models_pytorch.__version__
    )
    monkeypatch.setattr(guard, "packages_distributions", lambda: {"cv2": providers})
    with pytest.raises(RuntimeError, match="one OpenCV provider"):
        guard.main()


@pytest.mark.parametrize("problem", [None, "missing", "empty"])
def test_installed_gate_requires_third_party_license_texts(tmp_path, problem):
    metadata = tmp_path / "cellseg_models_pytorch-0.1.30.dist-info"
    legal = metadata / "licenses"
    files = [Path("LICENSE"), Path("NOTICE.md")]
    files += [
        Path("LICENSES") / path.name
        for path in (Path(__file__).resolve().parents[1] / "LICENSES").glob("*.txt")
    ]
    for path in files:
        target = legal / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("retained notice\n")
    (metadata / "RECORD").write_text(
        "".join(f"{metadata.name}/licenses/{path},,\n" for path in files)
    )
    apache = legal / "LICENSES/Apache-2.0.txt"
    if problem == "missing":
        apache.unlink()
    elif problem == "empty":
        apache.write_text("")

    package = Distribution.at(metadata)
    if problem is None:
        guard.check_license_files(package)
    else:
        with pytest.raises(RuntimeError, match="Apache-2.0.txt"):
            guard.check_license_files(package)
