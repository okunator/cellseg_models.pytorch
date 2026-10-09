import subprocess
import sys


def test_wsi_import_without_matplotlib() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["matplotlib"] = None

import pytest

from cellseg_models_pytorch.wsi import SlideReader
from cellseg_models_pytorch.wsi.image import get_annotated_image

assert callable(SlideReader)
with pytest.raises(ImportError, match="Matplotlib is required.*pip install matplotlib"):
    get_annotated_image(None, [], 1.0)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
