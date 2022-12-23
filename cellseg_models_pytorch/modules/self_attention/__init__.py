from .attention_ops import compute_mha, mha, slice_mha
from .exact_attention import ExactSelfAttention
from .linformer import LinformerAttention

SELFATT_LOOKUP = {
    "exact": ExactSelfAttention,
    "linformer": LinformerAttention,
}

__all__ = [
    "ExactSelfAttention",
    "LinformerAttention",
    "mha",
    "slice_mha",
    "compute_mha",
    "SELFATT_LOOKUP",
]
