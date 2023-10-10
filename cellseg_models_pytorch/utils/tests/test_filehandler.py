from cellseg_models_pytorch.utils import FileHandler


def test_filehandler_get_gson(inst_map, type_map) -> None:
    xoff, yoff = FileHandler.get_xy_coords("x-5000_y-3000.geojson")
    assert xoff == 5000 and yoff == 3000

    gson = FileHandler.get_gson(
        inst_map, type_map, classes={"a": 0, "b": 1, "c": 2, "d": 3}
    )
    assert len(gson) == 7


def test_filehandler_read_h5(hdf5db):
    arrs = FileHandler.read_h5_patch(hdf5db, ix=1, return_type=True, return_sem=True)

    assert arrs["image"].any()
