import pytest
import torch

from cellseg_models_pytorch.utils.convolve import filter2D, gaussian, gaussian_kernel2d


@pytest.mark.parametrize("window_size", [5, 6])
def test_gaussian_kernel2d(window_size):
    kernel = gaussian_kernel2d(window_size, 2)
    assert kernel.shape == torch.Size([1, 1, window_size, window_size])


def test_convolve():
    im = torch.rand([1, 1, 14, 14])
    convolved = filter2D(im, gaussian_kernel2d(5, 2))
    assert convolved.shape == im.shape


@pytest.mark.parametrize("window_size", [3, 4])
def test_gaussian_values_and_options(window_size):
    positions = torch.tensor(
        [i - (window_size - 1) / 2 for i in range(window_size)],
        dtype=torch.float64,
    )
    expected = torch.exp(-(positions**2) / 8)
    expected /= expected.sum()
    vector = gaussian(window_size, 2, device=torch.device("cpu"), dtype=torch.float64)
    kernel = gaussian_kernel2d(
        window_size, 2, n_channels=2, device=torch.device("cpu"), dtype=torch.float64
    )
    torch.testing.assert_close(vector, expected)
    assert kernel.dtype == torch.float64
    assert kernel.device == torch.device("cpu")
    assert kernel.shape == (2, 1, window_size, window_size)
    for channel in kernel:
        torch.testing.assert_close(channel[0], expected[:, None] * expected[None, :])
    torch.testing.assert_close(
        kernel.sum(dim=(1, 2, 3)), torch.ones(2, dtype=torch.float64)
    )


@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda", 0), marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "dtype", [None, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_gaussian_kernel_preserves_requested_dtype_under_autocast(
    dtype, device, autocast_dtype
):
    expected_vector = gaussian(31, 26, device=device, dtype=dtype)
    expected = gaussian_kernel2d(31, 26, n_channels=2, device=device, dtype=dtype)

    with torch.autocast(device.type, dtype=autocast_dtype):
        actual_vector = gaussian(31, 26, device=device, dtype=dtype)
        actual = gaussian_kernel2d(31, 26, n_channels=2, device=device, dtype=dtype)

    for value, reference in ((actual_vector, expected_vector), (actual, expected)):
        assert value.dtype == (torch.float32 if dtype is None else dtype)
        assert value.device == device
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("window_size", [3, 4])
@pytest.mark.parametrize(
    "dtype", [None, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_gaussian_helpers_support_meta_device(window_size, dtype):
    device = torch.device("meta")
    vector = gaussian(window_size, 2, device=device, dtype=dtype)
    kernel = gaussian_kernel2d(
        window_size, 2, n_channels=2, device=device, dtype=dtype
    )

    assert vector.shape == (window_size,)
    assert kernel.shape == (2, 1, window_size, window_size)
    for value in (vector, kernel):
        assert value.device == device
        assert value.dtype == (torch.float32 if dtype is None else dtype)


@pytest.mark.parametrize("window_size", [3, 4])
def test_filter_values_channel_isolation_and_gradient(window_size):
    image = torch.arange(24, dtype=torch.float64).reshape(1, 2, 3, 4)
    image.requires_grad_()
    kernel = gaussian_kernel2d(window_size, 2, n_channels=2, dtype=torch.float64)
    result = filter2D(image, kernel)
    # Replicated borders: even kernels put the extra padding at the rear.
    front = (window_size - 1) // 2
    expected = torch.zeros_like(image)
    for channel in range(2):
        for row in range(3):
            for col in range(4):
                for kr in range(window_size):
                    for kc in range(window_size):
                        source_row = min(max(row + kr - front, 0), 2)
                        source_col = min(max(col + kc - front, 0), 3)
                        expected[0, channel, row, col] += (
                            image.detach()[0, channel, source_row, source_col]
                            * kernel[channel, 0, kr, kc]
                        )
    torch.testing.assert_close(result, expected)
    assert result.dtype == image.dtype
    assert result.device == image.device
    result.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert (image.grad > 0).all()
    torch.testing.assert_close(
        image.grad.sum(), torch.tensor(24.0, dtype=torch.float64)
    )
