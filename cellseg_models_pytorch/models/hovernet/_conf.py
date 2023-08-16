from typing import Any, Dict, Tuple

__all__ = ["_create_hovernet_args"]


def _create_hovernet_args(
    depth: int,
    n_dense_blocks: Tuple[int, int],
    norm: str,
    act: str,
    conv: str,
    att: str,
    preact: bool,
    preattend: bool,
    use_style: bool,
    merge_policy: str,
    skip_params: Dict[str, Any],
    upsampling: str,
) -> Tuple[Dict[str, Any], ...]:
    """Create the correct args to build HoVerNet architecture."""
    skip_params = skip_params if skip_params is not None else {"k": None}

    d12 = tuple(
        {
            "merge_policy": merge_policy,
            "upsampling": upsampling,
            "short_skips": ("basic", "dense", "basic"),
            "block_types": (
                ("basic",),
                ("hover_dense",) * ndb,
                ("basic",),
            ),
            "kernel_sizes": ((3,), (3,) * ndb, (1,)),
            "expand_ratios": ((0.5,), (0.0625,) * ndb, (1.0,)),
            "squeeze_ratio": 2.0,
            "groups": ((1,), (4,) * ndb, (1,)),
            "biases": ((False,), (False,) * ndb, (False,)),
            "normalizations": ((None,), (norm,) * ndb, (None,)),
            "activations": ((None,), (act,) * ndb, (None,)),
            "convolutions": ((conv,), (conv,) * ndb, (conv,)),
            "attentions": ((None,), (None,) * ndb, (att,)),
            "preactivates": ((False,), (preact,) * ndb, (False,)),
            "preattends": ((False,), (False,) * ndb, (preattend,)),
            "use_styles": ((False,), (use_style,) * ndb, (False,)),
            "skip_params": {
                "short_skips": ("basic",),
                "block_types": (("basic",),),
                "convolutions": ((conv,),),
                "normalizations": ((norm,),),
                "activations": ((act,),),
                **skip_params,
            },
        }
        for ndb in n_dense_blocks
    )

    d3 = (
        {
            "merge_policy": merge_policy,
            "upsampling": upsampling,
            "short_skips": ("basic",),
            "block_types": (("basic",),),
            "kernel_sizes": ((3,),),
            "expand_ratios": ((1.0,),),
            "squeeze_ratio": 1.0,
            "groups": ((1,),),
            "biases": ((False,),),
            "normalizations": ((None,),),
            "activations": ((None,),),
            "convolutions": ((conv,),),
            "attentions": ((att,),),
            "preactivates": ((False,),),
            "preattends": ((preattend,),),
            "use_styles": ((use_style,),),
            "skip_params": {
                "short_skips": ("basic",),
                "block_types": (("basic",),),
                "convolutions": ((conv,),),
                "normalizations": ((norm,),),
                "activations": ((act,),),
                **skip_params,
            },
        },
    )

    d45 = (
        {
            "merge_policy": merge_policy,
            "upsampling": upsampling,
            "short_skips": ("basic",),
            "block_types": (("basic",),),
            "kernel_sizes": ((3,),),
            "expand_ratios": ((1.0,),),
            "squeeze_ratio": 1.0,
            "groups": ((1,),),
            "biases": ((False,),),
            "normalizations": ((norm,),),
            "activations": ((act,),),
            "convolutions": ((conv,),),
            "attentions": ((att,),),
            "preactivates": ((preact,),),
            "preattends": ((preattend,),),
            "use_styles": ((use_style,),),
            "skip_params": {
                "short_skips": ("basic",),
                "block_types": (("basic",),),
                "convolutions": ((conv,),),
                "normalizations": ((norm,),),
                "activations": ((act,),),
                **skip_params,
            },
        },
    ) * (depth - 3)

    return d12 + d3 + d45
