import pytest

from cellseg_models_pytorch.models import cellpose_base
from cellseg_models_pytorch.optimizers import OPTIM_LOOKUP, adjust_optim_params


@pytest.mark.parametrize("optim", ["adam"])
def test_optim_setup(optim):
    model = cellpose_base(type_classes=3)
    optim_params = {
        "encoder": {"lr": 0.004, "weight_decay": 0.3},
        "decoder": {"lr": 0.003, "weight_decay": 0.2},
    }
    params = adjust_optim_params(model, optim_params)
    optimizer = OPTIM_LOOKUP[optim](params)

    assert all([p_g["lr"] == p["lr"] for p_g, p in zip(optimizer.param_groups, params)])
    assert all(
        [
            p_g["weight_decay"] == p["weight_decay"]
            for p_g, p in zip(optimizer.param_groups, params)
        ]
    )
