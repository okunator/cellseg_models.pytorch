"""Reject overlapping OpenCV distributions in the installed-package gate."""

import sys
from pathlib import Path

import pytest

from tools import check_installed_package as guard


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
