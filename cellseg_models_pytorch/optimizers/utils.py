from typing import Dict, List

import torch.nn as nn


def adjust_optim_params(
    model: nn.Module,
    encoder_lr: float,
    encoder_wd: float,
    decoder_lr: float,
    decoder_wd: float,
    remove_bias_wd: bool = True,
) -> List[Dict[str, Dict]]:
    """Adjust the learning parameters for optimizer.

    1. Adjust learning rate and weight decay in the pre-trained
        encoder and decoders.
    2. Remove weight decay from bias terms to reduce overfitting.

    "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    - https://arxiv.org/pdf/1812.01187

    Parameters
    ----------
        model : nn.Module
            The encoder-decoder segmentation model.
        encoder_lr : float
            Learning rate of the model encoder.
        encoder_wd : float
            Weight decay for the model encoder.
        decoder_lr : float
            Learning rate of the model decoder.
        decoder_wd : float
            Weight decay for the model decoder.
        remove_bias_wd : bool, default=True
            If True, the weight decay from the bias terms is removed from the model
            params. Ignored if `remove_wd`=True.

    Returns
    -------
        List[Dict[str, Dict]]:
            a list of kwargs (str, Dict pairs) containing the model params.
    """
    params = list(model.named_parameters())
    encoder_params = {"encoder": {"lr": encoder_lr, "weight_decay": encoder_wd}}
    decoder_params = {"decoder": {"lr": decoder_lr, "weight_decay": decoder_wd}}

    adjust_params = []
    for name, parameters in params:
        opts = {}
        for enc, enc_opts in encoder_params.items():
            if enc in name:
                for key, item in enc_opts.items():
                    opts[key] = item

        for dec, dec_opts in decoder_params.items():
            if dec in name:
                for key, item in dec_opts.items():
                    opts[key] = item

        if remove_bias_wd:
            if name.endswith("bias"):
                opts["weight_decay"] = 0.0

        adjust_params.append({"params": parameters, **opts})

    return adjust_params
