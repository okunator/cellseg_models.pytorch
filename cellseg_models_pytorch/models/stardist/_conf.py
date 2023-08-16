from typing import Any, Dict, Tuple

__all__ = ["_create_stardist_args"]


def _create_stardist_args(
    depth: int,
    norm: str,
    act: str,
    conv: str,
    att: str,
    preact: bool,
    preattend: bool,
    short_skip: str,
    use_style: bool,
    block_type: str,
    merge_policy: str,
    skip_params: Dict[str, Any],
    upsampling: str,
) -> Tuple[Dict[str, Any], ...]:
    """Create the args to build CellPose-Unet architecture."""
    skip_params = skip_params if skip_params is not None else {"k": None}

    return (
        {
            "merge_policy": merge_policy,
            "upsampling": upsampling,
            "short_skips": (short_skip,),
            "block_types": ((block_type, block_type),),
            "kernel_sizes": ((3, 3),),
            "expand_ratios": ((1.0, 1.0),),
            "groups": ((1, 1),),
            "biases": ((False, False),),
            "normalizations": ((norm, norm),),
            "activations": ((act, act),),
            "convolutions": ((conv, conv),),
            "attentions": ((att, att),),
            "preactivates": ((preact, preact),),
            "preattends": ((preattend, preattend),),
            "use_styles": ((use_style, False),),
            "skip_params": {
                "short_skips": (short_skip,),
                "block_types": ((block_type,),),
                "convolutions": ((conv,),),
                "normalizations": ((norm,),),
                "activations": ((act,),),
                **skip_params,
            },
        },
    ) * depth
