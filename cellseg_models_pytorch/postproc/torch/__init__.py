"""Post-processing functions that retain tensors on their input device."""

from .flow import follow_flows_torch

__all__ = ["follow_flows_torch"]
