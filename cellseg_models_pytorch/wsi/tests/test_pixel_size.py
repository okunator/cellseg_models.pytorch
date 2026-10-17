from types import SimpleNamespace

import numpy as np
import pytest

from cellseg_models_pytorch.wsi import bioio_reader, cucim_reader, openslide_reader
from cellseg_models_pytorch.wsi._base_reader import _pixel_size_um
from cellseg_models_pytorch.wsi.reader import SlideReader


@pytest.mark.parametrize(
    "values,units",
    [
        (("0.25", "0.5"), ("µm", "µm")),
        ((0.25, 0.5), ("micrometer", "micrometer")),
        ((250, 500), ("nm", "nm")),
        ((0.00025, 0.0005), ("mm", "mm")),
        ((250, 0.5), ("nanometer", "um")),
    ],
)
def test_pixel_size_converts_units_and_preserves_both_axes(values, units):
    actual = _pixel_size_um(values, units)
    assert actual == pytest.approx((0.25, 0.5))
    assert all(isinstance(value, float) for value in actual)


@pytest.mark.parametrize(
    "values",
    [
        (None, 0.5),
        (0.25, None),
        ("bad", 0.5),
        (0, 0.5),
        (-0.25, 0.5),
        (np.nan, 0.5),
        (0.25, np.inf),
        (0.25,),
        (0.25, 0.5, 1),
    ],
)
def test_missing_or_invalid_pixel_size_is_not_guessed(values):
    with pytest.raises(ValueError, match="Pixel"):
        _pixel_size_um(values)


@pytest.mark.parametrize("units", [("pixel", "pixel"), ("", "µm"), ("µm",)])
def test_unknown_or_missing_pixel_units_are_rejected(units):
    with pytest.raises(ValueError, match="Pixel"):
        _pixel_size_um((0.25, 0.5), units)


@pytest.mark.parametrize("backend", ["OPENSLIDE", "CUCIM", "BIOIO"])
def test_slide_reader_exposes_backend_calibration_without_axis_swapping(
    tmp_path, monkeypatch, backend
):
    path = tmp_path / "slide.svs"
    path.touch()
    native = SimpleNamespace(
        properties={"openslide.mpp-y": "0.25", "openslide.mpp-x": "0.5"},
        level_dimensions=((32, 16), (16, 8)),
        resolutions={"level_dimensions": ((32, 16), (16, 8))},
        spacing=lambda axes: [250, 500] if axes == "YX" else pytest.fail(axes),
        spacing_units=lambda axes: ["nm", "nm"] if axes == "YX" else pytest.fail(axes),
        resolution_level_dims={0: (16, 32), 1: (8, 16)},
        current_scene_index=1,
        ome_metadata=SimpleNamespace(
            images=[
                SimpleNamespace(pixels=None),
                SimpleNamespace(
                    pixels=SimpleNamespace(
                        physical_size_y=250,
                        physical_size_x=0.5,
                        physical_size_y_unit=SimpleNamespace(value="nm"),
                        physical_size_x_unit=SimpleNamespace(value="µm"),
                    )
                ),
            ]
        ),
    )
    monkeypatch.setattr(openslide_reader, "HAS_OPENSLIDE", True)
    monkeypatch.setattr(
        openslide_reader,
        "openslide",
        SimpleNamespace(OpenSlide=lambda path: native),
        raising=False,
    )
    monkeypatch.setattr(cucim_reader, "HAS_CUCIM", True)
    monkeypatch.setattr(cucim_reader, "CuImage", lambda path: native, raising=False)
    monkeypatch.setattr(bioio_reader, "HAS_BIOIO", True)
    monkeypatch.setattr(bioio_reader, "BioImage", lambda path: native, raising=False)

    reader = SlideReader(path, backend=backend)

    assert reader.dimensions == (16, 32)
    assert reader.pixel_size == pytest.approx((0.25, 0.5))
    assert reader.level_downsamples[1] == (2, 2)


def test_bioio_reports_unavailable_calibration_instead_of_assuming_units():
    class NativeReader:
        @property
        def ome_metadata(self):
            raise NotImplementedError("No OME metadata transformer")

    reader = bioio_reader.BioIOReader.__new__(bioio_reader.BioIOReader)
    reader._BioIOReader__reader = NativeReader()
    with pytest.raises(ValueError, match="calibrated OME metadata"):
        _ = reader.pixel_size


@pytest.mark.parametrize("mpp", ["0.2520", None, "bad", "0", "nan"])
def test_cucim_aperio_calibration_when_generic_tiff_spacing_is_missing(mpp):
    reader = cucim_reader.CucimReader.__new__(cucim_reader.CucimReader)
    reader._CucimReader__reader = SimpleNamespace(
        spacing=lambda axes: [1, 1],
        spacing_units=lambda axes: ["", ""],
        metadata={"aperio": {"MPP": mpp}},
    )
    if mpp == "0.2520":
        assert reader.pixel_size == (0.252, 0.252)
    else:
        with pytest.raises(ValueError, match="Pixel"):
            _ = reader.pixel_size


def test_cucim_keeps_calibrated_anisotropic_spacing_instead_of_aperio_mpp():
    reader = cucim_reader.CucimReader.__new__(cucim_reader.CucimReader)
    reader._CucimReader__reader = SimpleNamespace(
        spacing=lambda axes: [0.25, 0.5],
        spacing_units=lambda axes: ["µm", "µm"],
        metadata={"aperio": {"MPP": "0.2520"}},
    )
    assert reader.pixel_size == (0.25, 0.5)
