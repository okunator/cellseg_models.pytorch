import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InstanSegCriterion"]


class InstanSegCriterion(nn.Module):
    """Train InstanSeg seed scores and object-center offsets.

    Call this criterion directly with pred and target; the generic Loss and
    MultiTaskLoss wrappers use yhat rather than pred and cannot call it unchanged.
    """

    def __init__(self, w_seed: float = 1.0, w_coord: float = 1.0) -> None:
        """Choose the contributions of seed and coordinate losses.

        Args:
            w_seed: Multiplier on binary seed classification loss.
            w_coord: Multiplier on mean squared coordinate-offset loss.
        """

        super().__init__()
        self.w_seed = w_seed
        self.w_coord = w_coord

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare InstanSeg reconstruction maps with instance labels.

        Args:
            pred: Floating (B, 4, H, W) maps on the target device. Channels 0 and 1
                encode x/y center offsets and channel 3 supplies seed logits.
                Channel 2 does not contribute to this criterion.
            target: Integer (B, H, W) instance labels. Zero is background; positive
                IDs identify objects. Objects below five pixels get zero coordinate
                targets. The coordinate mean includes background pixels as well.

        Returns:
            Weighted sum of the mean seed and coordinate losses, a differentiable
            scalar on the prediction device. Center offsets are measured relative
            to width/height; this is not a cell-type classification loss.
        """

        B, C, H, W = pred.shape
        seed_map = pred[:, 3]
        coord_pred = pred[:, :2]

        seed_target = (target > 0).float()
        seed_loss = F.binary_cross_entropy_with_logits(seed_map, seed_target)

        coord_target = self._compute_coord_targets(target, H, W, pred.device)
        coord_pred_scaled = (torch.sigmoid(coord_pred) - 0.5) * 8
        coord_loss = F.mse_loss(coord_pred_scaled, coord_target)

        return self.w_seed * seed_loss + self.w_coord * coord_loss

    @staticmethod
    def _compute_coord_targets(
        target: torch.Tensor, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        B = target.shape[0]
        coord_target = torch.zeros(B, 2, H, W, device=device)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float),
            torch.arange(W, device=device, dtype=torch.float),
            indexing="ij",
        )
        for b in range(B):
            inst = target[b]
            uniq = inst.unique()
            uniq = uniq[uniq > 0]
            for uid in uniq:
                mask = inst == uid
                if mask.sum() < 5:
                    continue
                cy = yy[mask].mean()
                cx = xx[mask].mean()
                coord_target[b, 0, mask] = (cx - xx[mask]) / W
                coord_target[b, 1, mask] = (cy - yy[mask]) / H
        return coord_target
