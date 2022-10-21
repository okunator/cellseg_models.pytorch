from cellseg_models_pytorch.utils import FileHandler


def test_filehandler(inst_map, type_map) -> None:
    split_dict = FileHandler.get_offset("x-5000_y-3000")
    assert "x" in split_dict.keys() and "y" in split_dict.keys()
    assert 3000 in split_dict.values() and 5000 in split_dict.values()

    gson = FileHandler.get_gson(
        inst_map, type_map, classes={"a": 0, "b": 1, "c": 2, "d": 3}
    )
    assert len(gson) == 7
