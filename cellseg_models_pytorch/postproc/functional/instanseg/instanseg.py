from typing import Tuple

import numpy as np
import torch

from cellseg_models_pytorch.utils.instanseg import (
    _compute_crops_wrapper,
    _spatial_embedding,
    _torch_peak_local_max,
)

__all__ = ["post_proc_instanseg"]


def _convert(
    prob_input: torch.Tensor,
    coords_input: torch.Tensor,
    size: Tuple[int, int],
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    all_labels = torch.arange(
        1, 1 + prob_input.shape[0], dtype=torch.float32, device=prob_input.device
    )
    labels = torch.ones_like(prob_input) * torch.reshape(all_labels, (-1, 1, 1, 1))

    labels = labels.flatten()
    prob = prob_input.flatten()
    x = coords_input[0, ...].flatten()
    y = coords_input[1, ...].flatten()

    if size is None:
        size = (int(y.max() + 1), int(x.max() + 1))

    inds_prob = prob >= mask_threshold
    n_thresholded = torch.count_nonzero(inds_prob)
    if n_thresholded == 0:
        return torch.zeros(size, dtype=torch.float32, device=labels.device)

    arr = torch.zeros(
        (int(n_thresholded), 5), dtype=coords_input.dtype, device=labels.device
    )
    arr[:, 1] = y[inds_prob]
    arr[:, 2] = x[inds_prob]
    arr[:, 0] = arr[:, 2] * size[1] + arr[:, 1]
    arr[:, 3] = labels[inds_prob]

    inds_sorted = prob[inds_prob].argsort(descending=True, stable=True)
    arr = arr[inds_sorted, :]

    inds_sorted = arr[:, 0].argsort(descending=False, stable=True)
    arr = arr[inds_sorted, :]

    inds_unique = torch.ones_like(arr[:, 0], dtype=torch.bool)
    inds_unique[1:] = arr[1:, 0] != arr[:-1, 0]

    output = torch.zeros(size, dtype=torch.float32, device=labels.device)
    output[arr[inds_unique, 2], arr[inds_unique, 1]] = arr[inds_unique, 3].float()

    return output


def _remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


def _fast_sparse_iou(sparse_onehot: torch.Tensor) -> torch.Tensor:
    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,))[None].to_dense()
    union = sparse_sum.T + sparse_sum - intersection
    return intersection / union


def _find_connected_components(
    adjacency_matrix: torch.Tensor,
    max_iterations: int = 100,
) -> torch.Tensor:
    n = adjacency_matrix.shape[0]
    if n == 0:
        return torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long)

    labels = torch.arange(1, n + 1, device=adjacency_matrix.device, dtype=torch.long)
    M = adjacency_matrix + torch.eye(n, device=adjacency_matrix.device)
    indices = torch.nonzero(M)
    if indices.size(0) == 0:
        raise ValueError("Graph has no edges or self-loops")
    row = indices[:, 0]
    col = indices[:, 1]

    for _ in range(max_iterations):
        prev_labels = labels.clone()
        min_labels = torch.full(
            (n,), float("inf"), device=adjacency_matrix.device, dtype=torch.float
        )
        min_labels.scatter_reduce_(0, row, labels[col].float(), reduce="amin")
        labels = torch.minimum(labels, min_labels)
        if torch.equal(labels, prev_labels):
            break

    node_indices = torch.arange(
        1, n + 1, device=adjacency_matrix.device, dtype=torch.long
    )
    tentative_remapping = torch.stack((node_indices, labels))
    remapping = torch.cat(
        (
            torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long),
            tentative_remapping,
        ),
        dim=1,
    )
    return remapping


def _merge_sparse_predictions(
    x: torch.Tensor,
    coords: torch.Tensor,
    mask_map: torch.Tensor,
    size: list,
    mask_threshold: float = 0.5,
    window_size: int = 128,
    min_size: int = 10,
    overlap_threshold: float = 0.5,
    mean_threshold: float = 0.5,
) -> torch.Tensor:
    labels = _convert(
        x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold
    )[None]

    idx = torch.arange(1, size[0] + 1, device=x.device, dtype=coords.dtype)
    stack_ID = torch.ones(
        (size[0], window_size, window_size), device=x.device, dtype=coords.dtype
    )
    stack_ID = stack_ID * (idx[:, None, None] - 1)

    coords_flat = torch.stack((stack_ID.flatten(), coords[0] * size[2] + coords[1])).to(
        coords.dtype
    )

    fg = x.flatten() > mask_threshold
    x_flat = x.flatten()[fg]
    coords_flat = coords_flat[:, fg]
    mask_map_flat = mask_map.flatten()

    using_mps = False
    device = x_flat.device
    if x_flat.is_mps:
        using_mps = True
        device = torch.device("cpu")
        x_flat = x_flat.to(device)
        mask_map_flat = mask_map_flat.to(device)
        coords_flat = coords_flat.to(device)

    sparse_onehot = torch.sparse_coo_tensor(
        coords_flat,
        (x_flat > mask_threshold).float(),
        size=(size[0], size[1] * size[2]),
        dtype=x_flat.dtype,
        device=device,
        requires_grad=False,
    )

    # Empty seed crops still need an entry, otherwise later object IDs shift.
    object_areas = torch.sparse.sum(sparse_onehot, dim=1).to_dense()
    sum_mask_value = torch.sparse.sum(
        (sparse_onehot * mask_map_flat[None]), dim=1
    ).to_dense()
    mean_mask_value = sum_mask_value / object_areas.clamp_min(1)
    objects_to_remove = ~torch.logical_and(
        mean_mask_value > mean_threshold, object_areas > min_size
    )

    iou = _fast_sparse_iou(sparse_onehot)
    remapping = _find_connected_components((iou > overlap_threshold).float())

    if using_mps:
        remapping = remapping.to(x.device)
        labels = labels.to(x.device)

    labels = _remap_values(remapping, labels)

    labels_to_remove = (
        torch.arange(
            0,
            len(objects_to_remove),
            device=objects_to_remove.device,
            dtype=coords.dtype,
        )
        + 1
    )[objects_to_remove]

    labels[torch.isin(labels, labels_to_remove)] = 0

    return labels


@torch.no_grad()
def post_proc_instanseg(
    inst_map: np.ndarray,
    aux_map: np.ndarray,
    pixel_classifier: torch.nn.Module | None = None,
    mask_threshold: float = 0.53,
    peak_distance: int = 4,
    seed_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
    mean_threshold: float = -10000.0,
    window_size: int = 128,
    min_size: int = 10,
    max_seeds: int = 2000,
    **kwargs: object,
) -> np.ndarray:
    """Reconstruct nuclei from four raw InstanSeg maps and the trained classifier.

    Args:
        inst_map: Unused HW map, retained for the common reconstruction interface.
        aux_map: Floating (4, H, W) NumPy maps: XY offsets, learned shape feature,
            and seed logits. H and W must be at least two pixels.
        pixel_classifier: The network's trained probability_net. Computation uses
            its device and returns labels to CPU; its mode and weights are unchanged.
        mask_threshold: Membership probability threshold for each nucleus crop.
        peak_distance: Radius in pixels for finding seed maxima.
        seed_threshold: Seed probability threshold, after sigmoid.
        overlap_threshold: IoU threshold for joining crops of the same nucleus.
        mean_threshold: Minimum mean seed probability inside an object.
        window_size: Maximum even crop width in pixels, clipped to image size.
        min_size: Objects must have more than this many pixels.
        max_seeds: Return an empty mask if more seeds are found.
        **kwargs: Unused common reconstruction options.

    Returns:
        Int32 HW instance labels, with zero background and positive nucleus IDs.

    Raises:
        ValueError: The classifier is missing, maps are not 4HW, or crop/peak
            settings are invalid.
    """
    if pixel_classifier is None:
        raise ValueError("InstanSeg requires the network's trained pixel_classifier")
    if aux_map.ndim != 3 or aux_map.shape[0] != 4 or min(aux_map.shape[1:]) < 2:
        raise ValueError("InstanSeg aux_map must have shape (4, H, W), H/W >= 2")
    if window_size < 2 or window_size % 2 or peak_distance < 0:
        raise ValueError(
            "window_size must be positive and even; peak_distance must be nonnegative"
        )
    parameter = next(pixel_classifier.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    # NumPy flips can have negative strides, which Torch cannot read directly.
    aux_map_t = torch.as_tensor(
        np.ascontiguousarray(aux_map), device=device, dtype=dtype
    )
    if aux_map_t.dtype in (torch.float16, torch.bfloat16):
        aux_map_t = aux_map_t.float()
    height, width = aux_map_t.shape[1:]
    fields = _spatial_embedding(aux_map_t)
    sigma = aux_map_t[2:3]
    # Seed training uses BCE logits; thresholds refer to probabilities.
    mask_map = aux_map_t[3].sigmoid()

    local_centroids_idx = _torch_peak_local_max(
        mask_map, neighbourhood_size=int(peak_distance), minimum_value=seed_threshold
    )

    fields_at_centroids = fields[
        :, local_centroids_idx[:, 0], local_centroids_idx[:, 1]
    ]

    if local_centroids_idx.shape[0] > max_seeds or local_centroids_idx.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int32)

    window_size = min(window_size, height, width)
    window_size = window_size - window_size % 2

    h, w = height, width

    crops, coords = _compute_crops_wrapper(
        fields,
        fields_at_centroids.T,
        sigma,
        local_centroids_idx.int(),
        pixel_classifier,
        window_size,
    )

    coords = coords[1:]

    C = crops.shape[0]
    crops = torch.sigmoid(crops)
    label = _merge_sparse_predictions(
        crops,
        coords,
        mask_map,
        size=(C, h, w),
        mask_threshold=mask_threshold,
        window_size=window_size,
        min_size=min_size,
        overlap_threshold=overlap_threshold,
        mean_threshold=mean_threshold,
    ).int()

    return label.squeeze().cpu().numpy().astype(np.int32)
