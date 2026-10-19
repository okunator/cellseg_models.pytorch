"""Check standalone SSIM values, target interpretation and gradients."""

import numpy as np
import pytest
import torch
from scipy.signal import convolve2d

from cellseg_models_pytorch.losses import SSIM, Loss
from cellseg_models_pytorch.losses.criterions.ssim import ssim


def _maps(dtype=torch.float64):
    image = torch.linspace(0.1, 0.9, 256, dtype=dtype).reshape(1, 1, 16, 16)
    return image, image.roll(3, -1)


def _numpy_ssim(image, target, window_size=11, data_range=1):
    """Evaluate Gaussian windows independently, including replicated borders."""
    size = min(window_size, *image.shape[-2:])
    coordinates = np.arange(size) - (size - 1) / 2
    gaussian = np.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = np.outer(gaussian, gaussian)
    padding = ((size - 1) // 2, size // 2)

    def mean(values):
        return convolve2d(
            np.pad(values, (padding, padding), mode="edge"), kernel, mode="valid"
        )

    similarities, contrasts = [], []
    for batch, reference in zip(image, target):
        for x, y in zip(batch, reference):
            mx, my = mean(x), mean(y)
            vx, vy = mean(x * x) - mx**2, mean(y * y) - my**2
            covariance = mean(x * y) - mx * my
            contrast = (2 * covariance + (0.03 * data_range) ** 2) / (
                vx + vy + (0.03 * data_range) ** 2
            )
            luminance = (2 * mx * my + (0.01 * data_range) ** 2) / (
                mx**2 + my**2 + (0.01 * data_range) ** 2
            )
            similarities.append(luminance * contrast)
            contrasts.append(contrast)
    return np.stack(similarities).reshape(image.shape), np.mean(contrasts)


@pytest.mark.parametrize(
    "shape,window",
    [((2, 2, 16, 17), 11), ((1, 1, 4, 6), 11), ((1, 2, 7, 8), 4), ((1, 1, 1, 1), 11)],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_raw_similarity_matches_independent_gaussian_windows(shape, window, device):
    rng = np.random.default_rng(19)
    image, target = rng.uniform(0.1, 0.9, (2, *shape))
    expected, contrast = _numpy_ssim(image, target, window)
    actual, cs = ssim(
        torch.from_numpy(image).to(device),
        torch.from_numpy(target).to(device),
        window,
        1,
    )
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=1e-10, atol=1e-11)
    assert cs.item() == pytest.approx(contrast, abs=1e-11)


@pytest.mark.parametrize("values", [(0.5, 0.25), (0.9, 0.8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_constant_maps_have_analytic_loss_and_gradient(values, dtype, device):
    amplitude = torch.tensor(values[0], dtype=dtype, device=device, requires_grad=True)
    image = amplitude.expand(1, 1, 16, 16)
    target = torch.full_like(image, values[1])
    a, b = amplitude.item(), target[0, 0, 0, 0].item()
    # Flat images have no local contrast, so only brightness affects their SSIM.
    denominator = a**2 + b**2 + 0.01**2
    similarity = (2 * a * b + 0.01**2) / denominator
    derivative = -(b - a * similarity) / denominator
    tolerance = 2e-7 if dtype == torch.float32 else 1e-11
    loss, contrast = SSIM(return_cs=True, data_range=1)(image, target)
    assert loss.item() == pytest.approx((1 - similarity) / 2, abs=tolerance)
    assert contrast.item() == pytest.approx(0, abs=tolerance)
    loss.backward()
    assert amplitude.grad.item() == pytest.approx(derivative, abs=10 * tolerance)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_identical_maps_have_zero_loss_and_gradients(dtype, device):
    image = torch.rand(1, 2, 8, 9, generator=torch.Generator().manual_seed(23))
    image = image.to(device=device, dtype=dtype).requires_grad_()
    loss, contrast = SSIM(return_cs=True, data_range=1)(image, image.detach())
    assert loss.item() == 0
    assert contrast.item() == 0
    (loss + contrast).backward()
    torch.testing.assert_close(image.grad, torch.zeros_like(image), atol=1e-7, rtol=0)


@pytest.mark.parametrize("target_kind", ["bchw", "shared"])
def test_both_input_gradients_match_independent_finite_differences(target_kind):
    image, target = _maps()
    image = image.expand(1, 2, 16, 16).clone().requires_grad_()
    target = (
        (target[:, 0] if target_kind == "shared" else target.expand_as(image))
        .clone()
        .requires_grad_()
    )
    loss, contrast = SSIM(return_cs=True, data_range=1)(image, target)
    (loss + 0.2 * contrast).backward()
    arrays = [image.detach().numpy().copy(), target.detach().numpy().copy()]

    def objective():
        reference = arrays[1]
        if target_kind == "shared":
            reference = np.repeat(reference[:, None], 2, axis=1)
        similarity, cs = _numpy_ssim(arrays[0], reference)
        return np.clip(1 - similarity.mean(), 0, 1) / 2 + 0.2 * np.clip(1 - cs, 0, 1)

    # Nudge one pixel at a time to check gradients without relying on autograd.
    for array, tensor in zip(arrays, [image, target]):
        for index in [0, 15, 127, array.size - 1]:
            original = array.flat[index]
            array.flat[index] = original + 1e-5
            upper = objective()
            array.flat[index] = original - 1e-5
            lower = objective()
            array.flat[index] = original
            assert tensor.grad.flatten()[index].item() == pytest.approx(
                (upper - lower) / 2e-5, abs=2e-9
            )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_precision_values_and_gradients_match_double_reference(dtype, device):
    image, target = [x.to(dtype) for x in _maps()]
    prediction = image.to(device).clone().requires_grad_()
    reference = image.double().clone().requires_grad_()
    expected = SSIM(data_range=1, return_cs=True)(reference, target.double())
    actual = SSIM(data_range=1, return_cs=True)(prediction, target.to(device))
    raw, cs = _numpy_ssim(image.double().numpy(), target.double().numpy())
    assert expected[0].item() == pytest.approx((1 - raw.mean()) / 2, abs=1e-11)
    assert expected[1].item() == pytest.approx(1 - cs, abs=1e-11)
    for result, truth in zip(actual, expected):
        assert result.dtype == (
            torch.float64 if dtype == torch.float64 else torch.float32
        )
        torch.testing.assert_close(result.cpu().double(), truth, atol=1e-5, rtol=0)
    sum(actual).backward()
    sum(expected).backward()
    assert torch.isfinite(prediction.grad).all()
    torch.testing.assert_close(
        prediction.grad.cpu().double(), reference.grad, atol=8e-5, rtol=0.01
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_autocast_does_not_change_float32_values_or_gradients(dtype, device):
    image, target = [x.float().to(device) for x in _maps()]
    prediction = image.clone().requires_grad_()
    reference = image.clone().requires_grad_()
    expected = SSIM(return_cs=True, data_range=1)(reference, target)
    with torch.autocast(device_type=device, dtype=dtype):
        actual = SSIM(return_cs=True, data_range=1)(prediction, target)
    for value, truth in zip(actual, expected):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, truth, atol=0, rtol=0)
    sum(actual).backward()
    sum(expected).backward()
    # GPU sums can run in a different order, causing tiny rounding differences.
    torch.testing.assert_close(prediction.grad, reference.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_shared_fractional_targets_are_maps_and_preserve_promoted_gradients(dtype):
    image, target = _maps()
    prediction = image.float().expand(1, 3, 16, 16).clone().requires_grad_()
    target = target[:, 0].to(dtype).requires_grad_()
    double_target = target.detach().double().requires_grad_()
    expected = SSIM(data_range=1)(prediction.double(), double_target)
    actual = SSIM(data_range=1)(prediction, target)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual.double(), expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        target.grad.double(), double_target.grad, atol=1e-5, rtol=0.01
    )
    same = torch.full((1, 1, 5, 6), 0.25, dtype=dtype)
    assert SSIM(data_range=1)(same, same[:, 0]).item() == 0


def test_integer_targets_match_one_hot_maps():
    image = torch.rand(1, 3, 5, 6, generator=torch.Generator().manual_seed(7))
    labels = torch.arange(30).reshape(1, 5, 6) % 3
    maps = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
    torch.testing.assert_close(SSIM()(image, labels), SSIM()(image, maps))
    with pytest.raises(RuntimeError):
        SSIM()(image, labels - 1)
    with pytest.raises(RuntimeError):
        SSIM()(image, labels + 1)


def test_explicit_range_controls_symmetry_scaling_and_batch_independence():
    image, target = _maps()
    loss = SSIM(data_range=1)(image, target)
    torch.testing.assert_close(loss, SSIM(data_range=255)(image * 255, target * 255))
    byte_image = torch.linspace(0, 200, 256, dtype=torch.float64).reshape(1, 1, 16, 16)
    byte_target = torch.full_like(byte_image, 100)
    criterion = SSIM(data_range=255)
    value = criterion(byte_image, byte_target)
    torch.testing.assert_close(value, criterion(byte_target, byte_image))
    batched = criterion(
        torch.cat([byte_image, byte_image]), torch.cat([byte_target, byte_image])
    )
    torch.testing.assert_close(batched, value / 2)
    raw, _ = ssim(byte_image, byte_target, val_range=255)
    torch.testing.assert_close(value, (1 - raw.mean()) / 2)


@pytest.mark.parametrize(
    "offset,scale,expected_range",
    [(0, 1, 1), (-1, 1, 2), (0, 255, 255), (-1, 255, 256)],
)
def test_none_range_preserves_legacy_guessing(offset, scale, expected_range):
    image, target = _maps()
    image, target = offset + scale * (image - 0.1), offset + scale * (target - 0.1)
    torch.testing.assert_close(
        SSIM()(image, target), SSIM(data_range=expected_range)(image, target)
    )


def test_negative_similarity_keeps_legacy_loss_cap_and_return_contract():
    image = (torch.arange(256).reshape(1, 1, 16, 16) % 2).double().requires_grad_()
    loss, contrast = SSIM(return_cs=True, data_range=1)(image, 1 - image.detach())
    assert loss.item() == 0.5
    assert contrast.item() == 1
    loss.backward()
    assert torch.count_nonzero(image.grad).item() == 0
    maps = torch.zeros(1, 1, 4, 4)
    assert Loss(name="ssim", data_range=1)(maps, maps).item() == 0


@pytest.mark.parametrize("window", [0, -1, 2.5, True])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError, match="window_size"):
        SSIM(window)
    with pytest.raises(ValueError, match="window_size"):
        ssim(*_maps(), window_size=window)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_ranges_are_rejected(value):
    with pytest.raises(ValueError, match="data_range"):
        SSIM(data_range=value)
    with pytest.raises(ValueError, match="data_range"):
        ssim(*_maps(), val_range=value)


@pytest.mark.parametrize("shape", [(1, 8, 8), (0, 1, 8, 8), (1, 0, 8, 8), (1, 1, 0, 8)])
def test_invalid_prediction_shapes_are_rejected(shape):
    image = torch.zeros(shape)
    with pytest.raises(ValueError, match="BCHW"):
        SSIM()(image, image)


@pytest.mark.parametrize(
    "shape", [(2, 1, 8, 8), (1, 2, 8, 8), (1, 1, 7, 8), (2, 8, 8), (1, 8)]
)
def test_targets_cannot_broadcast_mismatched_axes(shape):
    with pytest.raises(ValueError, match="BCHW"):
        SSIM()(torch.zeros(1, 1, 8, 8), torch.zeros(shape))


def test_integer_maps_and_complex_targets_are_rejected():
    image = torch.zeros(1, 1, 8, 8)
    for prediction, target in [
        (image.long(), image),
        (image, image.long()),
        (image, image[:, 0].cfloat()),
    ]:
        with pytest.raises(TypeError, match="floating"):
            SSIM()(prediction, target)


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["ieee", "tf32", "none"])
@pytest.mark.parametrize("fail", [False, True])
def test_cuda_precision_settings_restored_on_success_and_failure(
    monkeypatch, precision, fail
):
    import importlib

    module = importlib.import_module("cellseg_models_pytorch.losses.criterions.ssim")
    image, target = [x.float().cuda() for x in _maps()]
    previous = torch.backends.cudnn.conv.fp32_precision
    previous_parent = torch.backends.cudnn.fp32_precision
    try:
        torch.backends.cudnn.fp32_precision = "tf32"
        # 'none' lets convolutions follow the parent setting instead of fixing their own.
        torch.backends.cudnn.conv.fp32_precision = precision
        if fail:

            def broken_kernel(*args, **kwargs):
                raise RuntimeError("injected kernel failure")

            monkeypatch.setattr(module, "gaussian_kernel2d", broken_kernel)
            with pytest.raises(RuntimeError, match="injected"):
                SSIM(data_range=1)(image, target)
        else:
            value = SSIM(data_range=1)(image, target)
            reference = SSIM(data_range=1)(image.cpu().double(), target.cpu().double())
            torch.testing.assert_close(
                value.cpu().double(), reference, atol=1e-5, rtol=0
            )
        assert torch.backends.cudnn.conv.fp32_precision == (
            "tf32" if precision == "none" else precision
        )
        # Later parent changes must still reach convolutions that inherit the setting.
        torch.backends.cudnn.fp32_precision = "ieee"
        assert torch.backends.cudnn.conv.fp32_precision == (
            "ieee" if precision == "none" else precision
        )
    finally:
        torch.backends.cudnn.fp32_precision = previous_parent
        torch.backends.cudnn.conv.fp32_precision = previous


@pytest.mark.cuda
def test_different_input_devices_are_rejected():
    image = torch.zeros(1, 1, 8, 8)
    with pytest.raises(ValueError, match="device"):
        SSIM()(image.cuda(), image)
