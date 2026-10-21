"""Small, offline checks for device-resident CellPose/Omnipose dynamics."""

import doctest
from pathlib import Path

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.postproc import (
    fill_holes_and_remove_small_masks,
    follow_flows_torch,
    get_masks_cellpose,
)
from cellseg_models_pytorch.postproc.functional.cellpose import integrator
from cellseg_models_pytorch.postproc.torch import flow as tensor_integrator
from cellseg_models_pytorch.utils import binarize, majority_vote_sequential

# Captured from the original NumPy integrator before introducing the tensor path.
REFERENCE = {
    False: [
        [0, 2, 4, 4],
        [2.200000047683715, 3, 5.074124955490232, 3.3593749409075833],
    ],
    True: [
        [0, 2, 4, 3.8087106636919605],
        [1.6250000248352687, 3, 5.6030132617924355, 4.042912562652754],
    ],
}


def _flows() -> np.ndarray:
    y, x = np.mgrid[:5, :7]
    return np.stack(((x - 3) * 0.2, (2 - y) * 0.3)).astype(np.float32)


@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_dynamics_match_preport_reference(
    suppress: bool, dtype: torch.dtype
) -> None:
    flows = torch.from_numpy(_flows()).to(dtype=dtype)
    mask = torch.ones((5, 7), dtype=torch.bool)
    endpoints, indices = follow_flows_torch(
        flows, mask, niter=4, suppress_euler=suppress
    )
    selected = endpoints[:, [0, 2, 4, 3], [1, 3, 6, 5]]
    torch.testing.assert_close(
        selected, torch.tensor(REFERENCE[suppress], dtype=torch.float32), rtol=0, atol=0
    )
    assert endpoints.dtype == torch.float32
    assert indices.dtype == torch.int64
    assert indices.shape == (35, 2)
    assert endpoints.device == indices.device == flows.device


@pytest.mark.parametrize("suppress", [False, True])
def test_numpy_interpolation_preserves_preport_values(suppress: bool) -> None:
    points = np.array([[0, 2, 4, 3], [1, 3, 6, 5]], dtype=np.float32)
    actual = integrator.steps2D_interp(points, _flows(), 4, suppress, use_gpu=False)
    assert actual.shape == (2, 4) and actual.dtype == np.float64
    np.testing.assert_allclose(actual, REFERENCE[suppress], rtol=0, atol=1e-14)


@pytest.mark.parametrize("count", [0, 1])
def test_legacy_numpy_interpolation_retains_original_shapes(count: int) -> None:
    points = np.array([[2], [3]], dtype=np.float32)[:, :count]
    actual = integrator.steps2D_interp(points, _flows(), niter=1, use_gpu=False)
    assert actual.shape == ((2, 0) if count == 0 else (2,))


def test_empty_flow_returns_grid_without_sampling(monkeypatch) -> None:
    def unexpected_sample(*args, **kwargs):
        pytest.fail("No seeds should not invoke sampling")

    monkeypatch.setattr(torch.nn.functional, "grid_sample", unexpected_sample)
    flows = torch.zeros((2, 5, 7))
    endpoints, indices = follow_flows_torch(flows)
    y, x = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
    torch.testing.assert_close(endpoints, torch.stack((y, x)).float(), rtol=0, atol=0)
    assert indices.shape == (0, 2)


def test_seed_threshold_union_and_unseeded_pixels() -> None:
    flows = torch.zeros((2, 3, 5), dtype=torch.float64)
    flows[0, 0, 1] = 4  # The legacy rule uses the x channel only.
    flows[1, 0, 2] = 1e-3
    flows[1, 1, 2] = 2e-3
    flows[1, 2, 4] = -3e-3
    mask = torch.zeros((3, 5), dtype=torch.bool)
    mask[0, 0] = True
    endpoints, indices = follow_flows_torch(flows, mask, niter=2)
    torch.testing.assert_close(indices, torch.tensor([[0, 0], [1, 2], [2, 4]]))
    torch.testing.assert_close(endpoints[:, 0, 1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(endpoints[:, 0, 2], torch.tensor([0.0, 2.0]))
    original, _ = follow_flows_torch(flows, mask, niter=0)
    y, x = torch.meshgrid(torch.arange(3), torch.arange(5), indexing="ij")
    torch.testing.assert_close(original, torch.stack((y, x)).float(), rtol=0, atol=0)


def test_single_seed_moves_in_tensor_path_and_legacy_behavior_is_retained() -> None:
    flows = torch.zeros((2, 5, 7))
    flows[1, 2, 3] = 0.75
    endpoints, indices = follow_flows_torch(flows, niter=1)
    torch.testing.assert_close(indices, torch.tensor([[2, 3]]))
    torch.testing.assert_close(
        endpoints[:, 2, 3], torch.tensor([2.0, 3.75]), rtol=0, atol=0
    )
    actual, seeds = integrator.follow_flows(flows.numpy(), niter=1, use_gpu=False)
    np.testing.assert_array_equal(actual[:, 2, 3], [2.0, 3.0])
    assert actual.dtype == np.float32 and seeds.dtype == np.int32


def test_integration_clamps_to_image_borders() -> None:
    flows = torch.empty((2, 5, 7))
    flows[0] = 100
    flows[1] = -100
    endpoints, _ = follow_flows_torch(flows, niter=1)
    torch.testing.assert_close(endpoints[0], torch.full((5, 7), 4.0))
    torch.testing.assert_close(endpoints[1], torch.zeros((5, 7)))


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_singleton_spatial_axes_sample_center(shape: tuple[int, int]) -> None:
    flows = torch.ones((2, *shape))
    endpoints, indices = follow_flows_torch(flows, niter=1)
    assert torch.isfinite(endpoints).all()
    assert indices.shape == (shape[0] * shape[1], 2)
    for axis, size in enumerate(shape):
        assert torch.all((endpoints[axis] >= 0) & (endpoints[axis] <= size - 1))
        if size == 1:
            assert torch.count_nonzero(endpoints[axis]) == 0
        else:
            index = (0, 2) if axis == 1 else (2, 0)
            assert endpoints[axis][index] == 3


def test_noncontiguous_inputs_autograd_and_autocast_are_preserved(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(8)
    flows = torch.rand((2, 10, 14), generator=generator)[:, ::2, ::2].requires_grad_()
    mask = torch.ones((10, 14), dtype=torch.bool)[::2, ::2]
    before = flows.detach().clone()
    mask_before = mask.clone()
    expected, expected_indices = follow_flows_torch(flows, mask, niter=3)

    def unexpected_transfer(*args, **kwargs):
        pytest.fail("Tensor integration attempted a host/NumPy conversion")

    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_transfer)
    monkeypatch.setattr(torch.Tensor, "numpy", unexpected_transfer)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, indices = follow_flows_torch(flows, mask, niter=3)
        assert torch.is_autocast_enabled("cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(flows, before, rtol=0, atol=0)
    torch.testing.assert_close(mask, mask_before)
    assert flows.requires_grad and flows.grad is None
    assert not actual.requires_grad and not actual.is_inference()
    assert torch.is_grad_enabled()


@pytest.mark.parametrize("shape", [(2, 0, 3), (1, 3, 3), (2, 3), (2, 3, 3, 3)])
def test_invalid_flow_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        follow_flows_torch(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float16, torch.bfloat16])
def test_invalid_flow_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        follow_flows_torch(torch.zeros((2, 3, 5), dtype=dtype))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_flows_fail_instead_of_skipping_integration(value: float) -> None:
    flows = torch.zeros((2, 3, 5))
    flows[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        follow_flows_torch(flows)


@pytest.mark.parametrize("niter", [-1, 1.5, True])
def test_invalid_iteration_counts(niter) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        follow_flows_torch(torch.zeros((2, 3, 5)), niter=niter)


def test_invalid_seed_masks_and_device() -> None:
    flows = torch.zeros((2, 3, 5))
    with pytest.raises(TypeError, match="bool"):
        follow_flows_torch(flows, torch.zeros((3, 5)))
    for mask in (
        torch.zeros((3, 4), dtype=torch.bool),
        torch.empty((3, 5), device="meta", dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="shape and device"):
            follow_flows_torch(flows, mask)
    with pytest.raises(ValueError, match="CPU and CUDA"):
        follow_flows_torch(torch.empty((2, 3, 5), device="meta"))


def test_pinned_cellpose_reconstruction_preserves_masks() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/checkpoint_baseline/cellpose.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        classes = data["dense_1"].argmax(1)[0].astype(np.int32)
        mask = binarize(classes).astype(bool)
        flows = torch.from_numpy(data["dense_0"][0] * mask)
        endpoints, _ = follow_flows_torch(flows, torch.from_numpy(mask))
        actual = fill_holes_and_remove_small_masks(
            get_masks_cellpose(endpoints.numpy()), min_size=30
        ).astype(np.int32)
        reference = data["instance_mask"]
        np.testing.assert_array_equal(actual == 0, reference == 0)
        pairs = np.unique(np.stack((actual.ravel(), reference.ravel()), axis=1), axis=0)
        assert len(pairs) == len(np.unique(actual)) == len(np.unique(reference)) == 97
        np.testing.assert_array_equal(
            majority_vote_sequential(classes, actual).astype(np.int32),
            data["type_mask"],
        )


def test_tensor_flow_docstring_example() -> None:
    result = doctest.testmod(tensor_integrator)
    assert result.failed == 0 and result.attempted >= 4


def test_tensor_integration_propagates_kernel_errors(monkeypatch) -> None:
    def broken_kernel(*args, **kwargs):
        raise RuntimeError("integration failed")

    monkeypatch.setattr(tensor_integrator, "_steps2D_interp_torch", broken_kernel)
    with pytest.raises(RuntimeError, match="integration failed"):
        follow_flows_torch(torch.from_numpy(_flows()))


@pytest.mark.cuda
@pytest.mark.parametrize("suppress", [False, True])
def test_cuda_flow_integration_preserves_device_and_cpu_parity(
    suppress: bool, monkeypatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cpu_flows = torch.from_numpy(_flows())
    expected, expected_indices = follow_flows_torch(
        cpu_flows, niter=4, suppress_euler=suppress
    )
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    flows = cpu_flows.to(device)
    with monkeypatch.context() as patch:

        def unexpected_transfer(*args, **kwargs):
            pytest.fail("CUDA integration attempted a host/NumPy conversion")

        patch.setattr(torch.Tensor, "cpu", unexpected_transfer)
        patch.setattr(torch.Tensor, "numpy", unexpected_transfer)
        with torch.autocast("cuda", dtype=torch.float16):
            actual, indices = follow_flows_torch(
                flows, niter=4, suppress_euler=suppress
            )
    assert actual.device == indices.device == device
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(indices.cpu(), expected_indices)


def test_existing_tensor_import_path_is_preserved() -> None:
    assert integrator.follow_flows_torch is follow_flows_torch
    assert tensor_integrator.follow_flows_torch is follow_flows_torch


def test_legacy_functions_do_not_call_tensor_backend(monkeypatch) -> None:
    def unexpected_tensor_backend(*args, **kwargs):
        pytest.fail("Original NumPy integration must remain independent")

    monkeypatch.setattr(
        tensor_integrator, "_steps2D_interp_torch", unexpected_tensor_backend
    )
    actual, _ = integrator.follow_flows(_flows(), niter=4, use_gpu=False)
    np.testing.assert_array_equal(
        actual[:, [0, 2, 4, 3], [1, 3, 6, 5]],
        np.array(REFERENCE[False], dtype=np.float32),
    )
