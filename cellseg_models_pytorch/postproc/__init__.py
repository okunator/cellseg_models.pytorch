from .functional.cellpose._old_cellpose import (
    get_masks_cellpose_old,
    post_proc_cellpose_old,
)
from .functional.cellpose.cellpose import get_masks_cellpose, post_proc_cellpose
from .functional.cellpose.integrator import follow_flows_torch
from .functional.cellpose.utils import (
    fill_holes_and_remove_small_masks,
    gen_flows,
    normalize_field,
)
from .functional.hovernet import post_proc_hovernet
from .functional.omnipose import get_masks_omnipose, post_proc_omnipose
from .functional.stardist.nms import get_bboxes
from .functional.stardist.stardist import post_proc_stardist, post_proc_stardist_orig
from .functional.instanseg.instanseg import post_proc_instanseg

POSTPROC_LOOKUP = {
    "stardist_orig": post_proc_stardist_orig,
    "stardist": post_proc_stardist,
    "cellpose": post_proc_cellpose,
    "cellpose_old": post_proc_cellpose_old,
    "omnipose": post_proc_omnipose,
    "hovernet": post_proc_hovernet,
    "instanseg": post_proc_instanseg,
}

__all__ = [
    "POSTPROC_LOOKUP",
    "follow_flows_torch",
    "gen_flows",
    "fill_holes_and_remove_small_masks",
    "normalize_field",
    "get_masks_cellpose",
    "post_proc_cellpose",
    "get_masks_omnipose",
    "post_proc_omnipose",
    "post_proc_hovernet",
    "post_proc_stardist",
    "post_proc_stardist_orig",
    "get_masks_cellpose_old",
    "post_proc_cellpose_old",
    "get_bboxes",
    "post_proc_instanseg",
]
