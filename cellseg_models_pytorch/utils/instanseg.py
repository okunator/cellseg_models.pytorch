"""Shared InstanSeg embeddings, seed selection and instance crops."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _generate_coordinate_map(
    spatial_dim: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a (2, H, W) XY grid spanning width/4 and height/4, as upstream."""
    if spatial_dim != 2:
        raise ValueError("InstanSeg requires two spatial coordinates")
    xx = torch.linspace(0, width / 4, width, device=device, dtype=dtype)
    yy = torch.linspace(0, height / 4, height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((xx, yy))


def _spatial_embedding(pred: torch.Tensor) -> torch.Tensor:
    """Return 2HW XY embeddings from one raw 4HW output, preserving float64.

    Half/bfloat16 maps use float32 so adding the spatial grid keeps small offsets.
    """
    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    height, width = pred.shape[-2:]
    coordinates = _generate_coordinate_map(2, height, width, pred.device, pred.dtype)
    return (pred[:2].sigmoid() - 0.5) * 8 + coordinates


def _torch_peak_local_max(
    image: torch.Tensor,
    neighbourhood_size: int = 4,
    minimum_value: float = 0.5,
    return_map: bool = False,
    dtype: torch.dtype = torch.int,
) -> torch.Tensor:
    """Select one seed at each local maximum of an HW probability map."""
    height, width = image.shape
    pooled, indices = F.max_pool2d(
        image.reshape(1, 1, height, width),
        kernel_size=2 * neighbourhood_size + 1,
        stride=1,
        padding=neighbourhood_size,
        return_indices=True,
    )
    pixels = torch.arange(image.numel(), device=image.device).reshape_as(indices)
    peaks = (indices == pixels) & (pooled > minimum_value)
    if return_map:
        return peaks
    return torch.nonzero(peaks.reshape(height, width)).to(dtype)


def _centre_crop(
    centroids: torch.Tensor, window_size: int, h: int, w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixel indices and bounds for square crops kept inside the image."""
    # Move the crop, not the seed, when a nucleus is near an image edge.
    centers = centroids.clone()
    centers[:, 0].clamp_(window_size // 2, h - window_size // 2)
    centers[:, 1].clamp_(window_size // 2, w - window_size // 2)
    directions = torch.tensor([[-1, -1], [1, 1]], device=centroids.device)
    bounds = centers[:, None] + directions * (window_size // 2)

    axis = torch.arange(window_size, device=centroids.device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    pixels = torch.stack((yy, xx)).flatten(1)[:, None]
    pixels = pixels + bounds[:, 0].T[:, :, None]
    return pixels.flatten(1), bounds


def _feature_engineering(
    fields: torch.Tensor,
    centers: torch.Tensor,
    sigma: torch.Tensor,
    window_size: int,
    mesh_grid_flat: torch.Tensor,
) -> torch.Tensor:
    """Pair each crop pixel's seed-relative XY position with its learned feature."""
    count = centers.shape[0]
    samples = torch.cat((fields, sigma))[:, mesh_grid_flat[0], mesh_grid_flat[1]]
    samples = samples.reshape(3, count, (2 * window_size) ** 2).permute(1, 2, 0)
    offsets = samples[..., :2] - centers[:, None]
    return torch.cat((offsets, samples[..., 2:]), dim=-1).reshape(-1, 3)


def _compute_crops_wrapper(
    fields: torch.Tensor,
    centres: torch.Tensor,
    sigma: torch.Tensor,
    centroids_idx: torch.Tensor,
    pixel_classifier: nn.Module,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify crop pixels as belonging to their seed's nucleus.

    Args:
        fields: Floating 2HW XY embeddings on the classifier device.
        centres: Floating (N, 2) XY embeddings at the N seed pixels.
        sigma: Floating 1HW learned shape feature, on the same device.
        centroids_idx: Integer (N, 2) seed pixel positions in YX order.
        pixel_classifier: The network's shared three-input classifier.
        window_size: Even crop width, no larger than either image dimension.

    Returns:
        Membership logits (N, 1, window, window) and integer indices
        (3, N * window**2), ordered as crop number, Y and X. Crops stay inside
        the image; classifier precision follows its weights, with AMP disabled.
    """
    pixels, _ = _centre_crop(centroids_idx, window_size, *fields.shape[-2:])
    features = _feature_engineering(fields, centres, sigma, window_size // 2, pixels)
    parameter = next(pixel_classifier.parameters(), None)
    if parameter is not None:
        features = features.to(dtype=parameter.dtype)
    # This tiny classifier needs full precision when AMP is active for the encoder.
    with torch.autocast(device_type=fields.device.type, enabled=False):
        logits = pixel_classifier(features)
    count = centres.shape[0]
    logits = logits.reshape(count, 1, window_size, window_size)
    crop_ids = torch.arange(count, device=fields.device).repeat_interleave(
        window_size**2
    )
    return logits, torch.cat((crop_ids[None], pixels))
