"""Numerical and reconstruction checks for InstanSeg's shared training objective."""

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from cellseg_models_pytorch.losses import JointLoss, Loss, MultiTaskLoss
from cellseg_models_pytorch.losses.criterions.instanseg import (
    InstanSegCriterion,
    _lovasz_hinge,
)
from cellseg_models_pytorch.models.instanseg import instanseg_nuclei
from cellseg_models_pytorch.postproc.functional.instanseg.instanseg import (
    post_proc_instanseg,
)
from cellseg_models_pytorch.utils.instanseg import (
    _compute_crops_wrapper,
    _spatial_embedding,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def _example(device="cpu", dtype=torch.float64):
    values = torch.linspace(-0.7, 0.9, 2 * 4 * 6 * 10, device=device, dtype=dtype)
    values = values.reshape(2, 4, 6, 10)
    values[:, 3] = -3
    values[0, 3, 2, 2] = 2.3
    values[0, 3, 3, 7] = 1.7
    values[1, 3, 0, 1] = 1.3
    labels = torch.zeros(2, 6, 10, device=device, dtype=torch.long)
    labels[0, 1:4, 1:4] = 7
    labels[0, 2:5, 6:9] = 91
    labels[1, :2, :3] = 455
    classifier = nn.Linear(3, 1, device=device, dtype=dtype)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.17, -0.23, 0.31]]))
        classifier.bias.fill_(0.11)
    return values.requires_grad_(), labels, classifier


def _reference_lovasz(logits, labels):
    # Build the IoU error by adding sorted mistakes one pixel at a time.
    labels = labels.to(logits.dtype)
    errors = 1 - logits * (2 * labels - 1)
    order = errors.argsort(descending=True)
    remaining = labels.sum()
    union = labels.sum()
    previous = logits.new_zeros(())
    result = logits.sum() * 0
    for index in order:
        remaining = remaining - labels[index]
        union = union + (1 - labels[index])
        error = 1 - remaining / union
        result = result + errors[index].clamp_min(0) * (error - previous)
        previous = error
    return result


def _reference_loss(values, labels, classifier, seeds, mask=None, w_seed=1, w_inst=1.5):
    height, width = values.shape[-2:]
    yy = (
        torch.arange(height, device=values.device, dtype=values.dtype)
        * height
        / (4 * (height - 1))
    )
    xx = (
        torch.arange(width, device=values.device, dtype=values.dtype)
        * width
        / (4 * (width - 1))
    )
    valid = labels >= 0
    if mask is not None:
        valid = valid & mask.bool()
    image_losses = []
    for index, positions in enumerate(seeds):
        logits = values[index, 3]
        foreground = (labels[index] > 0).to(values.dtype)
        seed = (
            logits.clamp_min(0)
            - logits * foreground
            + torch.log1p(torch.exp(-logits.abs()))
        )
        seed = seed[valid[index]].sum() / valid[index].sum().clamp_min(1)
        field_x = (values[index, 0].sigmoid() - 0.5) * 8 + xx[None]
        field_y = (values[index, 1].sigmoid() - 0.5) * 8 + yy[:, None]
        losses = []
        for row, col in positions:
            top = max(0, min(row - 2, height - 4))
            left = max(0, min(col - 2, width - 4))
            box = (slice(top, top + 4), slice(left, left + 4))
            dx = field_x[box] - field_x[row, col].detach()
            dy = field_y[box] - field_y[row, col].detach()
            features = torch.stack((dx, dy, values[index, 2][box]), dim=-1)
            membership = features @ classifier.weight.T + classifier.bias
            truth = labels[index][box] == labels[index, row, col]
            included = valid[index][box]
            losses.append(
                _reference_lovasz(membership[..., 0][included], truth[included])
            )
        instance = torch.stack(losses).mean() if losses else values[index].sum() * 0
        image_losses.append(w_seed * seed + w_inst * instance)
    return torch.stack(image_losses).mean()


@pytest.mark.parametrize("device", DEVICES)
def test_lovasz_hand_calculated_value_and_gradient(device):
    logits = torch.tensor(
        [0.2, -0.3, 0.4], device=device, dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=torch.float64)
    actual = _lovasz_hinge(logits, labels)
    assert actual.item() == pytest.approx(43 / 60)
    actual.backward()
    torch.testing.assert_close(logits.grad, logits.new_tensor([-0.5, 1 / 6, -1 / 3]))
    assert torch.autograd.gradcheck(lambda x: _lovasz_hinge(x, labels), (logits,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("masked", [False, True])
def test_values_and_all_gradients_match_independent_objective(device, masked):
    values, labels, classifier = _example(device)
    mask = torch.ones_like(labels)
    if masked:
        labels[0, 1, 2] = -1
        mask[0, 2, 1] = 0
        mask[1, 1, 1] = 0
    original_labels = labels.clone()
    criterion = InstanSegCriterion(
        pixel_classifier=classifier,
        window_size=4,
        peak_distance=1,
        w_seed=0.7,
        w_inst=1.3,
    )
    actual = criterion(values, labels, mask=mask)
    expected = _reference_loss(
        values, labels, classifier, [[(2, 2), (3, 7)], [(0, 1)]], mask, 0.7, 1.3
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (values, *classifier.parameters())
    actual_grads = torch.autograd.grad(actual, inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(actual_grad).all()
        assert torch.count_nonzero(actual_grad) > 0
    assert (actual_grads[0].abs().sum((0, 2, 3)) > 0).all()
    assert torch.equal(labels, original_labels)
    if masked:
        assert torch.count_nonzero(actual_grads[0][0, :, 1, 2]) == 0
        assert torch.count_nonzero(actual_grads[0][0, :, 2, 1]) == 0


def test_loss_distinguishes_neighbor_instances_and_is_invariant_to_id_numbers():
    values, labels, classifier = _example()
    criterion = InstanSegCriterion(
        pixel_classifier=classifier, window_size=4, peak_distance=1
    )
    original = criterion(values, labels)
    renamed = torch.where(labels == 7, 500_000, labels)
    renamed = torch.where(renamed == 91, 2, renamed)
    torch.testing.assert_close(original, criterion(values, renamed), rtol=0, atol=0)
    # Nearby nuclei stay negative examples in each other's membership crop.
    labels[0, 1:4, 3:5] = 18
    assert not torch.isclose(original, criterion(values, labels))


@pytest.mark.parametrize("device", DEVICES)
def test_direct_factory_joint_and_multitask_routes_agree(device):
    values, labels, classifier = _example(device)
    options = dict(pixel_classifier=classifier, window_size=4, peak_distance=1)
    direct = InstanSegCriterion(**options)
    routed = Loss("instanseg", **options)
    assert routed.loss.pixel_classifier is classifier
    expected = direct(pred=values, target=labels)
    results = [
        direct(yhat=values, target=labels),
        routed(values, labels),
        JointLoss([routed])(yhat=values, target=labels),
        MultiTaskLoss({"inst": routed})({"inst": values}, {"inst": labels}),
    ]
    expected_grad = torch.autograd.grad(expected, values, retain_graph=True)[0]
    for actual in results:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual, values, retain_graph=True)[0],
            expected_grad,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "case",
    [
        "background",
        "unannotated",
        "masked",
        "no_seeds",
        "all_foreground",
        "single_pixel",
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_empty_and_small_targets_have_defined_values_and_gradients(case, device):
    classifier = nn.Linear(3, 1, device=device, dtype=torch.float64)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.fill_(0.2)
    maps = torch.zeros(1, 4, 3, 5, device=device, dtype=torch.float64)
    labels = torch.zeros(1, 3, 5, device=device, dtype=torch.long)
    mask = None
    if case == "unannotated":
        labels.fill_(-1)
    if case == "masked":
        mask = torch.zeros_like(labels)
    if case in ("all_foreground", "no_seeds"):
        labels.fill_(1)
    if case in ("all_foreground", "single_pixel"):
        labels[0, 1, 2] = 1
        maps[0, 3, 1, 2] = 2
    maps.requires_grad_()
    criterion = InstanSegCriterion(pixel_classifier=classifier, window_size=2)
    actual = criterion(maps, labels, mask=mask)
    if case in ("masked", "unannotated"):
        assert actual.item() == 0
    elif case in ("background", "no_seeds"):
        assert actual.item() == pytest.approx(math.log(2))
    else:
        positive_count = 15 if case == "all_foreground" else 1
        seed = (14 * math.log(2) + math.log1p(math.exp(-2))) / 15
        membership = 0.8 if positive_count == 15 else 1.1
        assert actual.item() == pytest.approx(seed + 1.5 * membership)
    actual.backward()
    assert torch.isfinite(maps.grad).all()
    if case in ("masked", "unannotated"):
        assert torch.count_nonzero(maps.grad) == 0


def test_binary_mask_excludes_seed_selection_and_instance_pixels():
    values, labels, classifier = _example()
    mask = torch.ones_like(labels)
    mask[0, 2, 2] = 0
    criterion = InstanSegCriterion(
        pixel_classifier=classifier, window_size=4, peak_distance=1
    )
    actual = criterion(values, labels, mask=mask)
    expected = _reference_loss(values, labels, classifier, [[(3, 7)], [(0, 1)]], mask)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(values.grad[0, :, 2, 2]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_crop_memory_is_bounded_and_sampling_is_reproducible(device):
    values = torch.zeros(1, 4, 30, 50, device=device)
    values[0, 3, ::2, ::2] = 2
    values.requires_grad_()
    labels = torch.arange(1, 1501, device=device).reshape(1, 30, 50) * 1000
    classifier = nn.Linear(3, 1, device=device)
    seen = []
    handle = classifier.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape)
    )
    criterion = InstanSegCriterion(
        pixel_classifier=classifier, window_size=8, peak_distance=0, max_seeds=5
    )
    torch.manual_seed(14)
    first = criterion(values, labels)
    torch.manual_seed(14)
    second = criterion(values, labels)
    handle.remove()
    assert seen == [torch.Size([5 * 8 * 8, 3])] * 2
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    first.backward()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_low_precision_loss_and_gradients_match_float64(device, dtype, autocast):
    values, labels, classifier = _example(device, torch.float32)
    values = values.detach().to(dtype).requires_grad_()
    reference = values.detach().double().requires_grad_()
    reference_classifier = copy.deepcopy(classifier).double()
    criterion = InstanSegCriterion(
        pixel_classifier=classifier, window_size=4, peak_distance=1
    )
    expected = _reference_loss(
        reference, labels, reference_classifier, [[(2, 2), (3, 7)], [(0, 1)]]
    )
    expected.backward()
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        actual = criterion(values, labels)
    actual.backward()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        values.grad, reference.grad.to(dtype), atol=2e-7, rtol=2e-3
    )
    for parameter, truth in zip(
        classifier.parameters(), reference_classifier.parameters(), strict=True
    ):
        torch.testing.assert_close(
            parameter.grad.double(), truth.grad, atol=2e-6, rtol=2e-6
        )
    assert torch.isfinite(values.grad).all()
    assert (values.grad.abs().sum((0, 2, 3)) > 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_float16_extreme_seed_logits_and_classifier_features_remain_finite(device):
    classifier = nn.Linear(3, 1, device=device)
    with torch.no_grad():
        classifier.weight.fill_(0.1)
        classifier.bias.zero_()
    values = torch.full((1, 4, 3, 5), 65504, device=device, dtype=torch.float16)
    values[:, 3] = -65504
    values[0, 3, 1, 2] = 65504
    values.requires_grad_()
    labels = torch.ones(1, 3, 5, device=device, dtype=torch.long)
    actual = InstanSegCriterion(pixel_classifier=classifier, window_size=2)(
        values, labels
    )
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual)
    actual.backward()
    assert torch.isfinite(values.grad).all()
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in classifier.parameters()
    )


@pytest.mark.parametrize("device", DEVICES)
def test_shared_crops_match_reconstruction_grid_on_non_square_border_inputs(device):
    maps = torch.zeros(4, 5, 9, device=device, dtype=torch.float64)
    maps[2] = torch.arange(45, device=device).reshape(5, 9) / 100
    classifier = nn.Linear(3, 1, device=device, dtype=torch.float64)
    with torch.no_grad():
        classifier.weight.copy_(maps.new_tensor([[0.2, -0.3, 0.5]]))
        classifier.bias.fill_(0.1)
    positions = torch.tensor([[0, 0], [4, 8]], device=device)
    fields = _spatial_embedding(maps)
    torch.testing.assert_close(
        fields[0, 0], torch.linspace(0, 9 / 4, 9, device=device, dtype=maps.dtype)
    )
    torch.testing.assert_close(
        fields[1, :, 0], torch.linspace(0, 5 / 4, 5, device=device, dtype=maps.dtype)
    )
    centers = fields[:, positions[:, 0], positions[:, 1]].T
    logits, coords = _compute_crops_wrapper(
        fields, centers, maps[2:3], positions, classifier, 4
    )
    assert logits.shape == (2, 1, 4, 4)
    for index, (top, left) in enumerate([(0, 0), (1, 5)]):
        yy, xx = torch.meshgrid(
            torch.arange(top, top + 4, device=device),
            torch.arange(left, left + 4, device=device),
            indexing="ij",
        )
        torch.testing.assert_close(
            coords[1:, index * 16 : (index + 1) * 16],
            torch.stack((yy.flatten(), xx.flatten())),
        )
        dx = xx.to(maps.dtype) * (9 / 32) - centers[index, 0]
        dy = yy.to(maps.dtype) * (5 / 16) - centers[index, 1]
        expected = dx * 0.2 - dy * 0.3 + maps[2, yy, xx] * 0.5 + 0.1
        torch.testing.assert_close(logits[index, 0], expected)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruction_uses_seed_probabilities_and_shared_classifier(device):
    classifier = nn.Linear(3, 1, device=device)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        classifier.bias.zero_()
    maps = np.zeros((4, 8, 12), dtype=np.float32)
    maps[2:] = -5
    maps[2, 1:3, 1:3] = 5
    maps[2, 5:7, 9:11] = 5
    maps[3, 2, 2] = 0.2
    maps[3, 6, 10] = 0.2
    snapshot = {key: value.clone() for key, value in classifier.state_dict().items()}
    actual = post_proc_instanseg(
        np.zeros((8, 12)), maps, classifier, window_size=4, peak_distance=1, min_size=0
    )
    expected = np.zeros((8, 12), dtype=np.int32)
    expected[1:3, 1:3] = 1
    expected[5:7, 9:11] = 2
    np.testing.assert_array_equal(actual, expected)
    assert classifier.training
    for key, value in classifier.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("axes", [(1,), (2,), (1, 2)])
def test_flipped_numpy_maps_match_contiguous_reconstruction(device, axes):
    classifier = nn.Linear(3, 1, device=device)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        classifier.bias.zero_()
    maps = np.zeros((4, 8, 12), dtype=np.float32)
    maps[2:] = -5
    maps[2, 1:3, 1:3] = 5
    maps[2, 5:7, 9:11] = 5
    maps[3, 2, 2] = 0.2
    maps[3, 6, 10] = 0.2
    flipped = np.flip(maps, axis=axes)
    before = flipped.copy()
    options = dict(window_size=4, peak_distance=1, min_size=0)
    expected = post_proc_instanseg(
        np.zeros((8, 12)), np.ascontiguousarray(flipped), classifier, **options
    )
    actual = post_proc_instanseg(np.zeros((8, 12)), flipped, classifier, **options)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual > 0, flipped[2] > 0)
    assert np.unique(actual).tolist() == [0, 1, 2]
    np.testing.assert_array_equal(flipped, before)


@pytest.mark.parametrize(
    "device,autocast",
    [
        ("cpu", False),
        ("cpu", True),
        pytest.param("cuda", False, marks=pytest.mark.cuda),
        pytest.param("cuda", True, marks=pytest.mark.cuda),
    ],
)
def test_real_model_trains_all_instance_channels_and_classifier_without_checkpoint_changes(
    device,
    autocast,
):
    torch.manual_seed(8)
    network = instanseg_nuclei(3, enc_name="resnet18", enc_pretrain=False).to(device)
    with torch.no_grad():
        for parameter in network.probability_net.parameters():
            parameter.fill_(0.05)
    keys = set(network.state_dict())
    criterion = InstanSegCriterion(
        pixel_classifier=network.probability_net, window_size=16, max_seeds=2
    )
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        outputs = network(torch.randn(1, 3, 32, 64, device=device))["nuc"].aux_map
    outputs.retain_grad()
    # Positive seed scores ensure the test exercises membership from the first step.
    maps = torch.cat((outputs[:, :3], outputs[:, 3:] + 3), dim=1)
    labels = torch.zeros(1, 32, 64, device=device, dtype=torch.long)
    labels[:, 8:24, 8:24] = 1
    before = [
        parameter.detach().clone() for parameter in network.probability_net.parameters()
    ]
    with torch.autocast(device, dtype=amp_dtype, enabled=autocast):
        loss = criterion(maps, labels)
    loss.backward()
    assert (outputs.grad.abs().sum((0, 2, 3)) > 0).all()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad)
        for parameter in network.probability_net.parameters()
    )
    optimizer.step()
    assert all(
        not torch.equal(parameter, old)
        for parameter, old in zip(
            network.probability_net.parameters(), before, strict=True
        )
    )
    assert set(network.state_dict()) == keys
    fresh = instanseg_nuclei(3, enc_name="resnet18", enc_pretrain=False)
    fresh.load_state_dict(network.state_dict(), strict=True)


@pytest.mark.parametrize(
    "options",
    [
        {"apply_sd": True},
        {"apply_ls": True},
        {"apply_svls": True},
        {"edge_weight": 2},
        {"class_weights": torch.ones(2)},
    ],
)
def test_factory_rejects_unsupported_generic_options(options):
    with pytest.raises(ValueError, match="generic"):
        Loss("instanseg", pixel_classifier=nn.Linear(3, 1), **options)


@pytest.mark.parametrize(
    "options,match",
    [
        ({}, "probability_net"),
        ({"window_size": 3}, "window_size"),
        ({"max_seeds": 0}, "max_seeds"),
        ({"peak_distance": -1}, "peak_distance"),
        ({"w_seed": -1}, "nonnegative"),
        ({"w_inst": float("nan")}, "finite"),
        ({"w_inst": 1, "w_coord": 1}, "not both"),
    ],
)
def test_invalid_constructor_options(options, match):
    settings = {"pixel_classifier": nn.Linear(3, 1), **options}
    if not options:
        settings.clear()
    with pytest.raises(ValueError, match=match):
        InstanSegCriterion(**settings)


def test_legacy_coordinate_weight_is_an_explicit_deprecated_instance_weight():
    classifier = nn.Linear(3, 1)
    with pytest.warns(DeprecationWarning, match="membership"):
        criterion = InstanSegCriterion(0.7, 0.3, pixel_classifier=classifier)
    assert criterion.w_seed == 0.7
    assert criterion.w_inst == 0.3


@pytest.mark.parametrize(
    "case,match",
    [
        ("channels", "B4HW"),
        ("spatial", "H/W"),
        ("integer_prediction", "floating"),
        ("target_shape", "BHW"),
        ("target_float", "integer"),
        ("mask_shape", "mask"),
        ("soft_mask", "binary"),
        ("target_weight", "target_weight"),
        ("both_names", "exactly one"),
        ("no_prediction", "exactly one"),
    ],
)
def test_invalid_inputs_raise_contextual_errors(case, match):
    values, labels, classifier = _example()
    arguments = {"pred": values, "target": labels}
    if case == "channels":
        arguments["pred"] = values[:, :3]
    elif case == "spatial":
        arguments["pred"] = values[:, :, :1]
    elif case == "integer_prediction":
        arguments["pred"] = values.long()
    elif case == "target_shape":
        arguments["target"] = labels[:, None]
    elif case == "target_float":
        arguments["target"] = labels.float()
    elif case == "mask_shape":
        arguments["mask"] = labels[:, None]
    elif case == "soft_mask":
        arguments["mask"] = torch.full_like(labels, 0.5, dtype=torch.float)
    elif case == "target_weight":
        arguments["target_weight"] = torch.ones_like(labels)
    elif case == "both_names":
        arguments["yhat"] = values
    elif case == "no_prediction":
        arguments.pop("pred")
    with pytest.raises(ValueError, match=match):
        InstanSegCriterion(pixel_classifier=classifier)(**arguments)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("w_seed,w_inst", [(0, 1.5), (0.7, 0), (0, 0)])
def test_disabled_contributions_have_zero_gradients(device, w_seed, w_inst):
    values, labels, classifier = _example(device)
    criterion = InstanSegCriterion(
        pixel_classifier=classifier,
        window_size=4,
        peak_distance=1,
        w_seed=w_seed,
        w_inst=w_inst,
    )
    actual = criterion(values, labels)
    expected = _reference_loss(
        values,
        labels,
        classifier,
        [[(2, 2), (3, 7)], [(0, 1)]],
        w_seed=w_seed,
        w_inst=w_inst,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    if w_seed == 0:
        assert torch.count_nonzero(values.grad[:, 3]) == 0
    if w_inst == 0:
        assert torch.count_nonzero(values.grad[:, :3]) == 0
        assert all(parameter.grad is None for parameter in classifier.parameters())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("min_size", [0, 5])
def test_empty_membership_crop_does_not_shift_later_object_filtering(device, min_size):
    classifier = nn.Linear(3, 1, device=device)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        classifier.bias.zero_()
    maps = np.zeros((4, 8, 12), dtype=np.float32)
    maps[2:] = -5
    maps[2, 5:7, 9:11] = 5
    maps[3, 2, 2] = 0.2
    maps[3, 6, 10] = 0.2
    actual = post_proc_instanseg(
        np.zeros((8, 12)),
        maps,
        classifier,
        window_size=4,
        peak_distance=1,
        min_size=min_size,
    )
    expected = np.zeros((8, 12), dtype=np.int32)
    if min_size == 0:
        expected[5:7, 9:11] = 2
    np.testing.assert_array_equal(actual, expected)
