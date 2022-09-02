import warnings

from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    ReduceLROnPlateau,
)

from .utils import adjust_optim_params

EXTRA_OPTIM_LOOKUP = {}
try:
    from torch_optimizer import (
        PID,
        QHM,
        SGDW,
        AccSGD,
        AdaBelief,
        AdaBound,
        AdaMod,
        AdamP,
        Apollo,
        DiffGrad,
        Lamb,
        Lookahead,
        NovoGrad,
        QHAdam,
        RAdam,
        Ranger,
        RangerQH,
        RangerVA,
        Yogi,
    )

    EXTRA_OPTIM_LOOKUP = {
        "accsgd": AccSGD,
        "adabound": AdaBound,
        "adabelief": AdaBelief,
        "adamp": AdamP,
        "apollo": Apollo,
        "adamod": AdaMod,
        "diffgrad": DiffGrad,
        "lamb": Lamb,
        "novograd": NovoGrad,
        "pid": PID,
        "qhadam": QHAdam,
        "qhm": QHM,
        "radam": RAdam,
        "sgwd": SGDW,
        "yogi": Yogi,
        "ranger": Ranger,
        "rangerqh": RangerQH,
        "rangerva": RangerVA,
        "lookahead": Lookahead,
    }
except ModuleNotFoundError:
    warnings.warn(
        "`torch_optimizer` optimzers not available. To use them, install with "
        "`pip install torch-optimizer`."
    )

SCHED_LOOKUP = {
    "lambda": LambdaLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "cyclic": CyclicLR,
    "exponential": ExponentialLR,
    "cosine_annealing": CosineAnnealingLR,
    "cosine_annealing_warm": CosineAnnealingWarmRestarts,
}

OPTIM_LOOKUP = {
    "adam": Adam,
    "rmsprop": RMSprop,
    "sgd": SGD,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "adamw": AdamW,
    "asgd": ASGD,
    **EXTRA_OPTIM_LOOKUP,
}

__all__ = [
    "SCHED_LOOKUP",
    "OPTIM_LOOKUP",
    "LambdaLR",
    "ReduceLROnPlateau",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "Adam",
    "RMSprop",
    "SGD",
    "Adadelta",
    "Adagrad",
    "Adamax",
    "AdamW",
    "ASGD",
    "AccSGD",
    "AdaBound",
    "AdaBelief",
    "AdamP",
    "Apollo",
    "AdaMod",
    "DiffGrad",
    "Lamb",
    "NovoGrad",
    "PID",
    "QHAdam",
    "QHM",
    "RAdam",
    "SGDW",
    "Yogi",
    "Ranger",
    "RangerQH",
    "RangerVA",
    "Lookahead",
    "adjust_optim_params",
]
