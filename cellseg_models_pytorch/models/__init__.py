import torch.nn as nn

from .base._multitask_unet import MultiTaskUnet
from .cellpose.cellpose import (
    CellPoseUnet,
    cellpose_base,
    cellpose_plus,
    omnipose_base,
    omnipose_plus,
)
from .cellvit.cellvit import (
    CellVitSAM,
    cellvit_sam_base,
    cellvit_sam_plus,
    cellvit_sam_small,
    cellvit_sam_small_plus,
)
from .hovernet.hovernet import (
    HoverNet,
    hovernet_base,
    hovernet_plus,
    hovernet_small,
    hovernet_small_plus,
)
from .stardist.stardist import (
    StarDistUnet,
    stardist_base,
    stardist_base_multiclass,
    stardist_plus,
)

MODEL_LOOKUP = {
    "cellpose_base": cellpose_base,
    "cellpose_plus": cellpose_plus,
    "omnipose_base": omnipose_base,
    "omnipose_plus": omnipose_plus,
    "hovernet_base": hovernet_base,
    "hovernet_plus": hovernet_plus,
    "hovernet_small": hovernet_small,
    "hovernet_small_plus": hovernet_small_plus,
    "stardist_base": stardist_base,
    "stardist_plus": stardist_plus,
    "stardist_base_multiclass": stardist_base_multiclass,
    "cellvit_sam_base": cellvit_sam_base,
    "cellvit_sam_plus": cellvit_sam_plus,
    "cellvit_sam_small": cellvit_sam_small,
    "cellvit_sam_small_plus": cellvit_sam_small_plus,
}


def get_model(
    name: str, type: str, ntypes: int = None, ntissues: int = None, **kwargs
) -> nn.Module:
    """Get the corect model at hand given name and type.

    Parameters
    ----------
        name : str
            Name of the model.
        type : str
            Type of the model. One of "base", "plus", "small", "small_plus".
        ntypes : int
            Number of cell types to segment.
        ntissues : int
            Number of tissue types to segment.
        **kwargs : dict
            Additional keyword arguments.

    Returns
    -------
        nn.Module: The specified model.
    """
    if name == "stardist":
        if type == "base":
            model = MODEL_LOOKUP["stardist_base_multiclass"](
                type_classes=ntypes, **kwargs
            )
        elif type == "plus":
            model = MODEL_LOOKUP["stardist_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
    elif name == "cellpose":
        if type == "base":
            model = MODEL_LOOKUP["cellpose_base"](type_classes=ntypes, **kwargs)
        elif type == "plus":
            model = MODEL_LOOKUP["cellpose_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
    elif name == "omnipose":
        if type == "base":
            model = MODEL_LOOKUP["omnipose_base"](type_classes=ntypes, **kwargs)
        elif type == "plus":
            model = MODEL_LOOKUP["omnipose_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
    elif name == "hovernet":
        if type == "base":
            model = MODEL_LOOKUP["hovernet_base"](type_classes=ntypes, **kwargs)
        elif type == "small":
            model = MODEL_LOOKUP["hovernet_small"](type_classes=ntypes, **kwargs)
        elif type == "plus":
            model = MODEL_LOOKUP["hovernet_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
        elif type == "small_plus":
            model = MODEL_LOOKUP["hovernet_small_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
    elif name == "cellvit":
        if type == "base":
            model = MODEL_LOOKUP["cellvit_sam_base"](type_classes=ntypes, **kwargs)
        elif type == "small":
            model = MODEL_LOOKUP["cellvit_sam_small"](type_classes=ntypes, **kwargs)
        elif type == "plus":
            model = MODEL_LOOKUP["cellvit_sam_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
        elif type == "small_plus":
            model = MODEL_LOOKUP["cellvit_sam_small_plus"](
                type_classes=ntypes, sem_classes=ntissues, **kwargs
            )
    else:
        raise ValueError("Unknown model type or name.")

    return model


__all__ = [
    "MultiTaskUnet",
    "HoverNet",
    "hovernet_base",
    "hovernet_plus",
    "hovernet_small",
    "hovernet_small_plus",
    "CellPoseUnet",
    "cellpose_base",
    "cellpose_plus",
    "omnipose_base",
    "omnipose_plus",
    "StarDistUnet",
    "stardist_base",
    "stardist_plus",
    "stardist_base_multiclass",
    "MODEL_LOOKUP",
    "get_model",
    "CellVitSAM",
    "cellvit_sam_base",
    "cellvit_sam_plus",
    "cellvit_sam_small",
    "cellvit_sam_small_plus",
]
