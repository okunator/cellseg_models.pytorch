import numpy as np
import pytest
import torch

from cellseg_models_pytorch.transforms.functional.normalization import minmax_normalize


@pytest.mark.parametrize("bounds", [(0, 1), (-1, 1), (3, 2)])
def test_tensor_minmax_handles_extreme_finite_float32_values(bounds):
    image = np.array([[-3e38, 0, 3e38]], dtype=np.float32)
    tensor = torch.from_numpy(image.copy())
    actual = minmax_normalize(tensor, *bounds)
    low, high = sorted(bounds)
    assert actual.dtype == torch.float32 and actual.device == tensor.device
    assert torch.isfinite(actual).all()
    np.testing.assert_allclose(
        actual.numpy(), [[low, (low + high) / 2, high]], atol=1e-7
    )
    np.testing.assert_allclose(
        actual.numpy(), minmax_normalize(image, *bounds), atol=5e-7
    )
    np.testing.assert_array_equal(tensor.numpy(), image)


@pytest.mark.cuda
def test_cuda_minmax_handles_extreme_finite_float32_values():
    image = torch.tensor([[-3e38, 0, 3e38]], device="cuda")
    actual = minmax_normalize(image)
    assert actual.is_cuda and actual.dtype == torch.float32
    torch.testing.assert_close(
        actual, torch.tensor([[0, 0.5, 1]], device="cuda"), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(image, torch.tensor([[-3e38, 0, 3e38]], device="cuda"))


def test_tensor_minmax_treats_sub_epsilon_ranges_as_constant_like_opencv():
    image = np.array([[0, 1e-30]], dtype=np.float32)
    np.testing.assert_array_equal(
        minmax_normalize(torch.from_numpy(image)).numpy(), minmax_normalize(image)
    )


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 3)])
@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("bounds", [(0, 1), (-1, 1), (3, 2)])
def test_tensor_minmax_matches_existing_opencv_contract(shape, constant, dtype, bounds):
    image = (
        np.full(shape, 27, dtype=dtype)
        if constant
        else np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    )
    tensor = torch.from_numpy(image.copy())
    expected = minmax_normalize(image, *bounds)
    actual = minmax_normalize(tensor, *bounds)
    assert actual.dtype == torch.float32 and actual.device == tensor.device
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=5e-7)
    np.testing.assert_array_equal(tensor.numpy(), image)


@pytest.mark.cuda
@pytest.mark.parametrize("constant", [False, True])
def test_cuda_minmax_preserves_device_and_global_channel_range(constant):
    image = (
        np.full((4, 5, 3), 27, dtype=np.uint8)
        if constant
        else np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    )
    actual = minmax_normalize(torch.as_tensor(image, device="cuda"))
    assert actual.is_cuda and actual.dtype == torch.float32
    np.testing.assert_allclose(
        actual.cpu().numpy(), minmax_normalize(image), rtol=0, atol=1e-7
    )
