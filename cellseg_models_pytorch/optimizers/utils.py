from typing import Dict, List

import torch.nn as nn


def adjust_optim_params(
    model: nn.Module, encoder_lr: float, encoder_wd: float, remove_bias_wd: bool = True
) -> List[Dict[str, Dict]]:
    """Adjust the model parameters for optimizer.

    1. Adjust learning rate and weight decay in the pre-trained
        encoder. Lower lr in encoder assumes that the encoder is
        already close to an optimum.
    2. Remove weight decay from bias terms to reduce overfitting

    "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    https://arxiv.org/pdf/1812.01187

    Parameters
    ----------
        model : nn.Module
            The encoder-decoder segmentation model.
        encoder_lr : float
            Learning rate of the model encoder.
        encoder_wd : float
            Weight decay for the model encoder.
        remove_bias_wd : bool, default=True
            Flag to whether to remove the weight decay from the bias terms.

    Returns
    -------
        List[Dict[str, Dict]]:
            a list of kwargs (str, Dict pairs) containing the model params.
    """
    params = list(model.named_parameters())
    encoder_params = {"encoder": dict(lr=encoder_lr, weight_decay=encoder_wd)}

    adjust_params = []
    for name, parameters in params:
        opts = {}
        for key, key_opts in encoder_params.items():
            if key in name:
                for k, i in key_opts.items():
                    opts[k] = i

        if remove_bias_wd:
            if name.endswith("bias"):
                opts["weight_decay"] = 0.0

        adjust_params.append({"params": parameters, **opts})

    return adjust_params
