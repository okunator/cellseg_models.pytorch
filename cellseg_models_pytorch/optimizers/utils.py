from typing import Any, Dict, List

import torch.nn as nn

__all__ = ["adjust_optim_params"]


def adjust_optim_params(
    model: nn.Module, optim_params: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Dict]]:
    """Adjust the learning parameters for optimizer.

    Parameters
    ----------
        model : nn.Module
            The encoder-decoder segmentation model.
        optim_params : Dict[str, Dict[str, Any]]
            optim params like learning rates, weight decays etc for diff parts of
            the network. E.g.
            {"encoder": {"weight_decay: 0.1, "lr":0.1}, "sem": {"lr": 0.1}}

    Returns
    -------
        List[Dict[str, Dict]]:
            a list of kwargs (str, Dict pairs) containing the model params.
    """
    params = list(model.named_parameters())

    adjust_params = []
    for name, parameters in params:
        opts = {}

        for block, block_params in optim_params.items():
            if block in name:
                for key, item in block_params.items():
                    opts[key] = item

        adjust_params.append({"params": parameters, **opts})

    return adjust_params
