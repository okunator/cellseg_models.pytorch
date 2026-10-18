import pytest


def test_normalization_exports_resolve_to_existing_classes():
    pytest.importorskip("albumentations")
    from cellseg_models_pytorch.transforms.albu_transforms import norm_transforms

    namespace = {}
    exec(
        "from cellseg_models_pytorch.transforms.albu_transforms.norm_transforms import *",
        namespace,
    )
    for name in ("MinMaxNormalization", "PercentileNormalization", "Normalization"):
        assert namespace[name] is getattr(norm_transforms, name)
