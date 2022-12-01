import pytest

from cellseg_models_pytorch.utils import run_pool


def wrap1(num):
    return num + 2


def wrap2(arg):
    return


@pytest.mark.parametrize(
    "typesets",
    [
        ("thread", "imap"),
        ("thread", "map"),
        ("thread", "uimap"),
        ("thread", "amap"),
        ("process", "map"),
        ("process", "amap"),
        ("process", "imap"),
        ("process", "uimap"),
        ("serial", "map"),
        ("serial", "imap"),
        pytest.param(("serial", "amap"), marks=pytest.mark.xfail),
        pytest.param(("serial", "uimap"), marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize("func", [wrap1, wrap2])
@pytest.mark.parametrize("ret", [True, False])
def test_run_pool(typesets, func, ret):
    args = [1, 2, 3, 4, 5, 6, 7, 8]
    res = run_pool(func, args, ret=ret, pooltype=typesets[0], maptype=typesets[1])

    if ret and func == wrap1:
        assert res == [3, 4, 5, 6, 7, 8, 9, 10]
    elif ret and func == wrap2:
        assert res == [None, None, None, None, None, None, None, None]
    else:
        # with amap, we always get a return val..
        if not ret and typesets[1] == "amap":
            pass
        else:
            assert res is None
