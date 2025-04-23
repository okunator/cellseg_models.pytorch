# Adapted from https://github.com/csccsccsccsc/cpp-net/blob/main/cppnet/models/SamplingFeatures2.py # noqa
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SamplingFeatures"]


def feature_sampling(
    feature_map: torch.Tensor,
    coord_map: torch.Tensor,
    nd_sampling: int,
    sampling_mode: str = "nearest",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample features from feature map with boundary-pixel coordinates.

    Parameters:
        feature_map (torch.Tensor):
            Input feature map. Shape: (B, C, H, W)
        coord_map (torch.Tensor):
            Boundary-pixel coordinate grid. Shape: (B, n_rays, 2, H, W)
        nd_sampling (int):
            Number of sampling points in each ray.
        sampling_mode (str, default="nearest"):
            Sampling mode, by default "nearest".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
            - sampled_features. Shape: (B, K*C', H, W)
            - sampling coords.
                Shape: (n_rays*B, H, W, 2) if nd_sampling > 0
                Shape: (B, n_rays*H, W, 2) if nd_sampling <= 0
    """
    b, c, h, w = feature_map.shape
    _, n_rays, _, _, _ = coord_map.shape

    sampling_coord = coord_map
    sampling_coord[:, :, 0, :, :] = sampling_coord[:, :, 0, :, :] / (w - 1)
    sampling_coord[:, :, 1, :, :] = sampling_coord[:, :, 1, :, :] / (h - 1)
    sampling_coord = sampling_coord * 2.0 - 1.0
    if n_rays * nd_sampling != c:
        raise ValueError(
            f"Number of rays ({n_rays}) * number of sampling points ({nd_sampling}) "
            f"should be equal to the number of channels ({c})"
        )

    if nd_sampling > 0:
        sampling_coord = sampling_coord.permute(1, 0, 3, 4, 2)
        sampling_coord = sampling_coord.flatten(start_dim=0, end_dim=1)  # kb, h, w, 2
        sampling_features = F.grid_sample(
            feature_map.view(b, n_rays, nd_sampling, h, w)
            .permute(1, 0, 2, 3, 4)
            .flatten(start_dim=0, end_dim=1),
            sampling_coord,
            mode=sampling_mode,
            align_corners=False,
        )  # kb, c', h, w
        sampling_features = sampling_features.view(
            n_rays, b, nd_sampling, h, w
        ).permute(1, 0, 2, 3, 4)  # b, k, c', h, w
    else:
        sampling_coord = sampling_coord.permute(0, 1, 3, 4, 2).flatten(
            start_dim=1, end_dim=2
        )  # b, kh, w, 2
        sampling_features = F.grid_sample(
            feature_map, sampling_coord, mode=sampling_mode, align_corners=False
        )
        sampling_features = sampling_features.view(b, c, n_rays, h, w).permute(
            0, 2, 1, 3, 4
        )  # b, k, c'/c, h, w

    sampling_features = sampling_features.flatten(
        start_dim=1, end_dim=2
    )  # b, k*c', h, w

    return sampling_features, sampling_coord


class SamplingFeatures(nn.Module):
    def __init__(self, n_rays: int, sampling_mode: str = "nearest") -> None:
        """Sample features from feature map with boundary-pixel coordinates.

        Parameters:
            n_rays (int)_
                Number of rays.
            sampling_mode (str, default="nearest"):
                Sampling mode, by default 'nearest'.
        """
        super().__init__()
        self.n_rays = n_rays
        self.angles = (
            torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0
        )  # 0 - 2*pi
        self.sin_angles = torch.sin(self.angles).view(1, n_rays, 1, 1)
        self.cos_angles = torch.cos(self.angles).view(1, n_rays, 1, 1)
        self.sampling_mode = sampling_mode

    def forward(
        self, feature_map: torch.Tensor, dist: torch.Tensor, nd_sampling: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample features and coords.

        Parameters:
            feature_map (torch.Tensor):
                Input feature map. Shape: (B, C, H, W)
            dist (torch.Tensor):
                Radial distance map. Shape: (B, n_rays, H, W)
            nd_sampling (int):
                Number of sampling points in each ray.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - sampled_features. Shape: (B, n_rays*C, H, W)
            - sampling coords.
                Shape: (n_rays*B, H, W, 2) if nd_sampling > 0
                Shape: (B, n_rays*H, W, 2) if nd_sampling <= 0
        """
        B, _, H, W = feature_map.shape

        if (
            self.sin_angles.device != dist.device
            or self.cos_angles.device != dist.device
        ):
            self.sin_angles = self.sin_angles.to(dist.device)
            self.cos_angles = self.cos_angles.to(dist.device)

        # sample radial coordinates (full circle) for the rays
        offset_ih = self.sin_angles * dist
        offset_iw = self.cos_angles * dist
        offsets = torch.stack([offset_iw, offset_ih], dim=2)  # (B, n_rays, 2, H, W)

        # create a flow/grid (for F.grid_sample)
        x_ = torch.arange(W).view(1, -1).expand(H, -1)
        y_ = torch.arange(H).view(-1, 1).expand(-1, W)
        grid = torch.stack([x_, y_], dim=0).float()

        # (B, 1, 2, H, W)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1).to(dist.device)

        # create the final offset grid
        offsets = offsets + grid

        sampled_features, sampling_coord = feature_sampling(
            feature_map, offsets, nd_sampling, self.sampling_mode
        )

        return sampled_features, sampling_coord, offsets
