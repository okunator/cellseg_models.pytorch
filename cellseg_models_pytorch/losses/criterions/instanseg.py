import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from cellseg_models_pytorch.utils.instanseg import (
    _compute_crops_wrapper,
    _spatial_embedding,
    _torch_peak_local_max,
)

__all__ = ["InstanSegCriterion"]


def _lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute binary Lovász hinge for the valid pixels in one instance crop."""
    if logits.numel() == 0:
        return logits.sum() * 0
    signs = 2 * labels - 1
    errors, order = (1 - logits * signs).sort(descending=True)
    foreground = labels[order]
    positives = foreground.sum()
    intersection = positives - foreground.cumsum(0)
    union = positives + (1 - foreground).cumsum(0)
    # Each sorted error is weighted by the extra IoU error it introduces.
    jaccard = 1 - intersection / union
    increments = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
    return torch.dot(F.relu(errors), increments)


class InstanSegCriterion(nn.Module):
    """Train seed scores and per-nucleus membership with the model's classifier.

    Uses upstream InstanSeg's seed BCE and per-crop Lovász hinge objective.
    Coordinate offsets and the learned feature train through membership, using
    the same spatial grid and classifier inputs as reconstruction.
    """

    def __init__(
        self,
        w_seed: float = 1.0,
        w_coord: float | None = None,
        *,
        pixel_classifier: nn.Module | None = None,
        w_inst: float | None = None,
        window_size: int = 128,
        max_seeds: int = 50,
        peak_distance: int = 4,
    ) -> None:
        """Choose the shared classifier and contributions of the two losses.

        Args:
            w_seed: Multiplier on mean binary seed cross-entropy.
            w_coord: Deprecated alias for w_inst; coordinate MSE is removed.
            pixel_classifier: The training network's probability_net, also used
                for reconstruction. Include it in the model optimizer; do not
                construct a separate classifier for the loss.
            w_inst: Multiplier on mean per-crop Lovász hinge. Defaults to 1.5.
            window_size: Maximum even crop width in pixels. Match reconstruction.
            max_seeds: Maximum sampled training crops per image. Random sampling
                bounds crop memory without a full-image mask per nucleus.
            peak_distance: Seed-search radius in pixels. Match reconstruction.

        Raises:
            ValueError: The shared classifier is missing or settings are invalid.

        Example:
            >>> from cellseg_models_pytorch.models.instanseg import instanseg_nuclei
            >>> network = instanseg_nuclei(3, enc_name="resnet18", enc_pretrain=False)
            >>> criterion = InstanSegCriterion(pixel_classifier=network.probability_net)
        """
        super().__init__()
        if not isinstance(pixel_classifier, nn.Module):
            raise ValueError("Pass the network's probability_net as pixel_classifier")
        if w_coord is not None:
            if w_inst is not None:
                raise ValueError("Use w_inst or the deprecated w_coord, not both")
            warnings.warn(
                "w_coord now weights instance membership; use w_inst instead",
                DeprecationWarning,
                stacklevel=2,
            )
            w_inst = w_coord
        if w_inst is None:
            w_inst = 1.5
        if any(not math.isfinite(weight) or weight < 0 for weight in (w_seed, w_inst)):
            raise ValueError("w_seed and w_inst must be finite and nonnegative")
        if not isinstance(window_size, int) or window_size < 2 or window_size % 2:
            raise ValueError("window_size must be an even integer >= 2")
        if not isinstance(max_seeds, int) or max_seeds < 1:
            raise ValueError("max_seeds must be a positive integer")
        if not isinstance(peak_distance, int) or peak_distance < 0:
            raise ValueError("peak_distance must be a nonnegative integer")
        self.pixel_classifier = pixel_classifier
        self.w_seed = w_seed
        self.w_inst = w_inst
        self.window_size = window_size
        self.max_seeds = max_seeds
        self.peak_distance = peak_distance

    def forward(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        *,
        yhat: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare raw InstanSeg maps with integer nucleus-instance labels.

        Args:
            pred: Floating (B, 4, H, W) maps: XY offsets, learned shape feature,
                seed logits. H/W >= 2. Keep the shared classifier on this device.
            target: Integer (B, H, W) labels on the same device. Zero is background,
                positive IDs identify nuclei, and negative labels are unannotated.
            yhat: Alternative name for pred, used by the composed loss helpers.
                Supply exactly one of pred and yhat.
            mask: Optional binary BHW inclusion mask. Excluded pixels contribute
                to neither loss nor seed selection, independently of their label.
            target_weight: Unsupported; passing a weight map raises ValueError.

        Returns:
            Mean over images of w_seed times seed BCE plus w_inst times mean
            per-seed membership loss. Empty/no-seed images have only seed loss;
            fully unannotated images contribute zero. Half/bfloat16 maps use
            float32 loss math; float64 is preserved. Classifier precision follows
            its parameters with autocast disabled for the small classifier.

        Raises:
            ValueError: Inputs violate the shape, device, label or mask contract.
        """
        if (pred is None) == (yhat is None):
            raise ValueError("Supply exactly one of pred and yhat")
        values = pred if pred is not None else yhat
        if values is None or target is None:
            raise ValueError("InstanSeg requires predictions and target")
        self._validate_inputs(values, target, mask, target_weight)
        if values.dtype in (torch.float16, torch.bfloat16):
            values = values.float()
        valid = target >= 0
        if mask is not None:
            valid = valid & mask.bool()
        total = values.sum() * 0
        for maps, labels, included in zip(values, target, valid, strict=True):
            seed_errors = F.binary_cross_entropy_with_logits(
                maps[3], (labels > 0).to(values.dtype), reduction="none"
            )
            seed_loss = (seed_errors * included).sum() / included.sum().clamp_min(1)
            instance_loss = self._instance_loss(maps, labels, included)
            total = total + self.w_seed * seed_loss + self.w_inst * instance_loss
        return total / values.shape[0]

    def _validate_inputs(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        target_weight: torch.Tensor | None,
    ) -> None:
        """Check instance labels and binary exclusions before selecting crops."""
        if (
            pred.ndim != 4
            or pred.shape[1] != 4
            or pred.shape[0] == 0
            or min(pred.shape[-2:]) < 2
            or not pred.is_floating_point()
        ):
            raise ValueError("InstanSeg predictions must be floating B4HW, H/W >= 2")
        if target.shape != (pred.shape[0], *pred.shape[-2:]):
            raise ValueError("InstanSeg target must be BHW matching predictions")
        if (
            target.is_floating_point()
            or target.is_complex()
            or target.dtype == torch.bool
        ):
            raise ValueError("InstanSeg target must contain integer instance IDs")
        if target.device != pred.device:
            raise ValueError("InstanSeg target and predictions must share a device")
        parameter = next(self.pixel_classifier.parameters(), None)
        if parameter is not None and parameter.device != pred.device:
            raise ValueError("pixel_classifier and predictions must share a device")
        if mask is not None:
            if mask.shape != target.shape or mask.device != pred.device:
                raise ValueError("mask must be BHW on the prediction device")
            if not ((mask == 0) | (mask == 1)).all():
                raise ValueError("InstanSeg mask must be binary")
        if target_weight is not None:
            raise ValueError("InstanSeg does not support target_weight maps")

    def _instance_loss(
        self, maps: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Classify pixels around selected foreground seeds as same-nucleus or not."""
        if self.w_inst == 0:
            return maps.sum() * 0
        # Only annotated nucleus pixels can supply training seeds.
        seed_scores = maps[3].detach().sigmoid() * ((labels > 0) & valid)
        seeds = _torch_peak_local_max(seed_scores, self.peak_distance, dtype=torch.long)
        if seeds.shape[0] == 0:
            return maps.sum() * 0
        if seeds.shape[0] > self.max_seeds:
            seeds = seeds[
                torch.randperm(seeds.shape[0], device=maps.device)[: self.max_seeds]
            ]
        fields = _spatial_embedding(maps)
        # Seeds are fixed references; surrounding pixels learn to belong to them.
        centers = fields[:, seeds[:, 0], seeds[:, 1]].T.detach()
        window_size = min(self.window_size, *maps.shape[-2:])
        window_size -= window_size % 2
        logits, coordinates = _compute_crops_wrapper(
            fields, centers, maps[2:3], seeds, self.pixel_classifier, window_size
        )
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        seed_ids = labels[seeds[:, 0], seeds[:, 1]]
        rows, cols = coordinates[1:]
        # A neighboring nucleus is a negative example, even though it is foreground.
        membership = labels[rows, cols].reshape(seeds.shape[0], -1) == seed_ids[:, None]
        included = valid[rows, cols].reshape(seeds.shape[0], -1)
        total = logits.sum() * 0
        for crop_logits, crop_labels, crop_valid in zip(
            logits.flatten(1), membership, included, strict=True
        ):
            total = total + _lovasz_hinge(
                crop_logits[crop_valid], crop_labels[crop_valid].to(logits.dtype)
            )
        return total / seeds.shape[0]
