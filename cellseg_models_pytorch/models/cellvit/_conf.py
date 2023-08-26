from typing import Any, Dict, Tuple

__all__ = ["_create_cellvit_args"]


def _create_cellvit_args(
    layer_depth: Tuple[int, ...],
    norm: str,
    act: str,
    conv: str,
    att: str,
    preact: bool,
    preattend: bool,
    short_skip: str,
    use_style: bool,
    merge_policy: str,
    skip_params: Dict[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Create the args to build CellVit-Unet decoders."""
    skip_params = skip_params if skip_params is not None else {"k": None}

    return tuple(
        {
            "layer_residual": False,
            "upsampling": "conv_transpose",
            "merge_policy": merge_policy,
            "short_skips": (short_skip,),
            "block_types": (("basic",) * ld,),
            "kernel_sizes": ((3,) * ld,),
            "expand_ratios": ((1.0,) * ld,),
            "groups": ((1,) * ld,),
            "biases": ((False,) * ld,),
            "normalizations": ((norm,) * ld,),
            "activations": ((act,) * ld,),
            "convolutions": ((conv,) * ld,),
            "attentions": ((att,) + (None,) * (ld - 1),),
            "preactivates": ((preact,) * ld,),
            "preattends": ((preattend,) * ld,),
            "use_styles": ((use_style,) * (ld - 1) + (False,),),
            "skip_params": {
                "short_skips": (short_skip,),
                "block_types": (("basic",),),
                "convolutions": ((conv,),),
                "normalizations": ((norm,),),
                "activations": ((act,),),
                **skip_params,
            },
        }
        for ld in layer_depth
    )
