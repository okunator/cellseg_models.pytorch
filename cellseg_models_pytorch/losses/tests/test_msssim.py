"""Check MS-SSIM values, scale reduction and differentiability independently."""

import numpy as np
import pytest
import torch
from scipy.signal import convolve2d

from cellseg_models_pytorch.losses import MSSSIM, SSIM, Loss


def _maps(height=33, width=49, dtype=torch.float64):
    y, x = torch.meshgrid(
        torch.linspace(0, 7, height, dtype=dtype),
        torch.linspace(0, 9, width, dtype=dtype),
        indexing="ij",
    )
    prediction = torch.stack(
        [0.5 + 0.2 * x.sin() * y.cos(), 0.4 + 0.25 * (x + y).cos()]
    ).unsqueeze(0)
    target = 0.65 * prediction.roll(3, -1) + 0.15
    return prediction, target


def _numpy_loss(prediction, target, window_size=3, data_range=1.0):
    """Evaluate spatial windows and five-scale products one image/channel at a time."""
    coordinates = np.arange(window_size) - window_size // 2
    gaussian = np.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = np.outer(gaussian, gaussian)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    scores = []
    for image, reference in zip(prediction, target):
        for x, y in zip(image, reference):
            score = 1.0
            for level, weight in enumerate(weights):
                mean_x = convolve2d(x, kernel, mode="valid")
                mean_y = convolve2d(y, kernel, mode="valid")
                variance_x = convolve2d(x * x, kernel, mode="valid") - mean_x**2
                variance_y = convolve2d(y * y, kernel, mode="valid") - mean_y**2
                covariance = convolve2d(x * y, kernel, mode="valid") - mean_x * mean_y
                contrast = (2 * covariance + (0.03 * data_range) ** 2) / (
                    variance_x + variance_y + (0.03 * data_range) ** 2
                )
                if level == 4:
                    luminance = (2 * mean_x * mean_y + (0.01 * data_range) ** 2) / (
                        mean_x**2 + mean_y**2 + (0.01 * data_range) ** 2
                    )
                    component = (contrast * luminance).mean()
                else:
                    component = contrast.mean()
                    height, width = x.shape
                    # Pooling drops the unused trailing padding on odd axes.
                    padding = ((height % 2, 0), (width % 2, 0))
                    x, y = np.pad(x, padding), np.pad(y, padding)
                    height, width = x.shape
                    x = x.reshape(height // 2, 2, width // 2, 2).mean((1, 3))
                    y = y.reshape(height // 2, 2, width // 2, 2).mean((1, 3))
                score *= np.clip(component, 0, 1) ** weight
            scores.append(score)
    return 1 - np.mean(scores)


@pytest.mark.parametrize("kind", ["zero", "constant", "random"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_default_identity_has_zero_loss_and_gradients(kind, dtype):
    generator = torch.Generator().manual_seed(19)
    image = torch.rand(1, 2, 161, 177, generator=generator).to(dtype)
    if kind == "zero":
        image.zero_()
    elif kind == "constant":
        image.fill_(0.5)
    image.requires_grad_()
    result = MSSSIM()(image, image.detach())
    assert result.item() == pytest.approx(0, abs=1e-7)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    result.backward()
    assert torch.isfinite(image.grad).all()
    torch.testing.assert_close(image.grad, torch.zeros_like(image), atol=1e-7, rtol=0)


def test_constant_maps_use_coarsest_luminance_once_and_have_analytic_gradient():
    amplitude = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    image = amplitude.expand(1, 1, 176, 176)
    target = torch.full_like(image, 0.25)
    similarity = (2 * 0.5 * 0.25 + 0.01**2) / (0.5**2 + 0.25**2 + 0.01**2)
    expected = 1 - similarity**0.1333
    derivative = (2 * 0.25 - 2 * 0.5 * similarity) / (0.5**2 + 0.25**2 + 0.01**2)
    expected_gradient = -0.1333 * similarity ** (0.1333 - 1) * derivative
    result = MSSSIM()(image, target)
    assert result.item() == pytest.approx(0.029296657839, abs=1e-11)
    assert result.item() == pytest.approx(expected, abs=1e-11)
    result.backward()
    assert amplitude.grad.item() == pytest.approx(expected_gradient, abs=1e-10)


@pytest.mark.parametrize("window_size,height,width", [(3, 33, 49), (11, 161, 177)])
def test_nonuniform_odd_maps_match_independent_spatial_reference(
    window_size, height, width
):
    prediction, target = _maps(height, width)
    result = MSSSIM(window_size)(prediction, target)
    expected = _numpy_loss(prediction.numpy(), target.numpy(), window_size)
    assert 0 < expected < 1
    assert result.item() == pytest.approx(expected, abs=1e-11)
    torch.testing.assert_close(result, MSSSIM(window_size)(target, prediction))


def test_spatial_gradients_match_independent_finite_differences():
    prediction, target = _maps()
    prediction.requires_grad_()
    MSSSIM(3)(prediction, target).backward()
    delta = 1e-5
    for index in [(0, 0, 0, 0), (0, 0, 15, 23), (0, 1, 16, 24), (0, 1, 32, 48)]:
        plus, minus = (
            prediction.detach().numpy().copy(),
            prediction.detach().numpy().copy(),
        )
        plus[index] += delta
        minus[index] -= delta
        expected = (
            _numpy_loss(plus, target.numpy()) - _numpy_loss(minus, target.numpy())
        ) / (2 * delta)
        assert abs(expected) > 1e-7
        assert prediction.grad[index].item() == pytest.approx(
            expected, abs=2e-9, rel=1e-5
        )


def test_prediction_and_target_gradcheck():
    prediction, target = _maps()
    scales = torch.tensor([0.9, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda values: MSSSIM(3)(prediction * values[0], target * values[1]),
        (scales,),
        atol=1e-6,
        rtol=1e-4,
    )


def test_batch_channel_scores_are_combined_before_averaging():
    prediction, target = _maps()
    prediction = torch.cat([prediction, prediction.flip(-2)], 0)
    target = torch.cat([prediction[:1], target.flip(-1)], 0)
    actual = MSSSIM(3)(prediction, target)
    separate = torch.stack(
        [
            MSSSIM(3)(prediction[b : b + 1, c : c + 1], target[b : b + 1, c : c + 1])
            for b in range(2)
            for c in range(2)
        ]
    ).mean()
    torch.testing.assert_close(actual, separate, atol=1e-12, rtol=0)
    assert actual.item() == pytest.approx(
        _numpy_loss(prediction.numpy(), target.numpy()), abs=1e-11
    )


def test_fixed_data_range_is_invariant_to_units():
    prediction, target = _maps()
    prediction.requires_grad_()
    normalized = MSSSIM(3)(prediction, target)
    byte_range = MSSSIM(3, data_range=255)(255 * prediction, 255 * target)
    torch.testing.assert_close(normalized, byte_range, atol=1e-12, rtol=0)
    normalized_gradient = torch.autograd.grad(normalized, prediction)[0]
    byte_gradient = torch.autograd.grad(byte_range, prediction)[0]
    torch.testing.assert_close(normalized_gradient, byte_gradient, atol=1e-12, rtol=0)


def test_negative_contrast_has_finite_zero_gradients():
    y, x = torch.meshgrid(torch.arange(48), torch.arange(48), indexing="ij")
    prediction = ((x + y) % 2).double().reshape(1, 1, 48, 48).requires_grad_()
    result = MSSSIM(3)(prediction, 1 - prediction.detach())
    assert result.item() == 1
    result.backward()
    torch.testing.assert_close(
        prediction.grad, torch.zeros_like(prediction), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_cpu_precision_and_autocast_match_float64_values_and_gradients(dtype, autocast):
    prediction, target = (item.to(dtype) for item in _maps())
    prediction.requires_grad_()
    reference = prediction.detach().double().requires_grad_()
    expected = MSSSIM(3)(reference, target.double())
    expected.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = MSSSIM(3)(prediction, target)
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(expected.item(), abs=1e-4)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm() > 0
    torch.testing.assert_close(
        prediction.grad.double(), reference.grad, atol=2e-6, rtol=0.02
    )


@pytest.mark.parametrize("window_size", [0, -1, 2, 2.5, True])
def test_invalid_windows_are_rejected(window_size):
    with pytest.raises(ValueError, match="positive odd integer"):
        MSSSIM(window_size)


@pytest.mark.parametrize("data_range", [0, -1, float("nan"), float("inf")])
def test_invalid_ranges_are_rejected(data_range):
    with pytest.raises(ValueError, match="positive and finite"):
        MSSSIM(data_range=data_range)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 161, 161),
        (0, 1, 161, 161),
        (1, 0, 161, 161),
        (1, 1, 160, 176),
        (1, 1, 176, 160),
    ],
)
def test_invalid_shapes_are_rejected(shape):
    image = torch.zeros(shape)
    with pytest.raises(ValueError, match="BCHW|at least 161"):
        MSSSIM()(image, image)


def test_mismatched_shapes_dtypes_and_integer_labels_are_rejected():
    image = torch.zeros(1, 1, 161, 161)
    with pytest.raises(ValueError, match="BCHW"):
        MSSSIM()(image, image[:, :, :-1])
    with pytest.raises(ValueError, match="same device and dtype"):
        MSSSIM()(image, image.double())
    with pytest.raises(TypeError, match="floating maps"):
        MSSSIM()(image.long(), image.long())


def test_single_pixel_window_still_requires_four_downsamplings():
    image = torch.zeros(1, 1, 16, 16)
    assert MSSSIM(1)(image, image).item() == 0
    with pytest.raises(ValueError, match="at least 16"):
        MSSSIM(1)(image[..., :15], image[..., :15])


def test_factory_keeps_unused_options_compatible_without_changing_the_loss():
    prediction, target = _maps()
    actual = Loss(
        "msssim",
        window_size=3,
        data_range=1,
        apply_sd=True,
        apply_ls=True,
        apply_svls=True,
        apply_mask=True,
        edge_weight=2,
        class_weights=torch.tensor([0.0, 5.0]),
    )(
        prediction,
        target,
        mask=torch.zeros_like(target[:, 0]),
        target_weight=torch.ones_like(target[:, 0]),
    )
    torch.testing.assert_close(actual, MSSSIM(3)(prediction, target))
    assert MSSSIM().state_dict() == {}


def test_standalone_ssim_retains_legacy_dissimilarity_and_contrast_return():
    image = torch.full((1, 1, 32, 32), 0.5)
    target = torch.full_like(image, 0.25)
    similarity = (2 * 0.5 * 0.25 + 0.01**2) / (0.5**2 + 0.25**2 + 0.01**2)
    loss, contrast = SSIM(return_cs=True)(image, target)
    assert loss.item() == pytest.approx((1 - similarity) / 2, abs=2e-5)
    assert contrast.item() == pytest.approx(0, abs=2e-5)
    assert SSIM()(image, image).item() == 0


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_precision_and_autocast_match_float64_values_and_gradients(
    dtype, autocast
):
    prediction, target = (
        item.to(device="cuda", dtype=dtype) for item in _maps(161, 177)
    )
    prediction.requires_grad_()
    reference = prediction.detach().cpu().double().requires_grad_()
    expected = MSSSIM()(reference, target.cpu().double())
    expected.backward()
    with torch.autocast(
        "cuda",
        dtype=dtype if dtype != torch.float32 else torch.float16,
        enabled=autocast,
    ):
        actual = MSSSIM()(prediction, target)
    assert actual.device == prediction.device
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(expected.item(), abs=1e-4)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm() > 0
    torch.testing.assert_close(
        prediction.grad.cpu().double(), reference.grad, atol=2e-6, rtol=0.02
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_identity_and_negative_contrast_gradients(dtype):
    prediction, _ = _maps(161, 177, dtype=dtype)
    prediction = prediction.cuda().requires_grad_()
    loss = MSSSIM()(prediction, prediction.detach())
    assert loss.item() == pytest.approx(0, abs=1e-7)
    loss.backward()
    torch.testing.assert_close(
        prediction.grad, torch.zeros_like(prediction), atol=1e-7, rtol=0
    )
    y, x = torch.meshgrid(
        torch.arange(48, device="cuda"), torch.arange(48, device="cuda"), indexing="ij"
    )
    checkerboard = ((x + y) % 2).to(dtype).reshape(1, 1, 48, 48).requires_grad_()
    loss = MSSSIM(3)(checkerboard, 1 - checkerboard.detach())
    assert loss.item() == 1
    loss.backward()
    torch.testing.assert_close(
        checkerboard.grad, torch.zeros_like(checkerboard), atol=0, rtol=0
    )


@pytest.mark.cuda
def test_different_devices_are_rejected():
    image = torch.zeros(1, 1, 161, 161)
    with pytest.raises(ValueError, match="same device and dtype"):
        MSSSIM()(image.cuda(), image)
